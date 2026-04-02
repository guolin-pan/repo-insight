/**
 * Chat interaction module — handles SSE streaming, message display, and Markdown rendering.
 *
 * Uses a typewriter queue to progressively display tokens that may arrive in bursts
 * from the LLM endpoint, creating a smooth real-time streaming experience.
 */

const ChatModule = {
    isStreaming: false,
    _abortController: null,

    // Smart auto-scroll state:
    // When AI is streaming, the scroll should follow the output — but only
    // if the user hasn't manually scrolled up to read earlier content.  Once
    // the user scrolls back down to the bottom, auto-scroll resumes.
    //
    // _userScrolledUp:  true when the user has scrolled away from the bottom.
    //                   Set by the "scroll" event listener on the messages
    //                   container.  Cleared when the user scrolls back to
    //                   the bottom (within a small tolerance).
    _userScrolledUp: false,

    // Typewriter state: queued text and animation control
    _tw: {
        queue: '',           // Characters waiting to be displayed
        displayed: '',       // Characters already rendered
        timer: null,         // requestAnimationFrame ID
        bubbleEl: null,      // Current bubble element
        charsPerFrame: 3,    // Characters to render per animation frame
        done: false,         // SSE stream finished
        onFinish: null,      // Callback when typewriter drains completely
    },

    // Reasoning (thinking) state:
    // Reasoning tokens arrive separately from content tokens via the
    // {"reasoning": "..."} SSE field.  They are accumulated and displayed
    // in a collapsible <details> panel above the main content bubble.
    // This gives the user visibility into the model's chain-of-thought
    // without cluttering the primary response area.
    _reasoning: {
        panelEl: null,       // The <details> element for reasoning display
        contentEl: null,     // The inner <div> that holds reasoning text
        text: '',            // Accumulated reasoning text
    },

    /** Start the typewriter animation loop. */
    _twStart(bubbleEl) {
        const tw = this._tw;
        tw.queue = '';
        tw.displayed = '';
        tw.bubbleEl = bubbleEl;
        tw.done = false;
        tw.onFinish = null;
        if (tw.timer) cancelAnimationFrame(tw.timer);
        tw.timer = null;
    },

    /** Push new text into the typewriter queue. */
    _twPush(text) {
        this._tw.queue += text;
        this._twSchedule();
    },

    /** Schedule the next render frame if not already scheduled. */
    _twSchedule() {
        if (this._tw.timer) return;
        this._tw.timer = requestAnimationFrame(() => this._twTick());
    },

    /** One animation tick: move characters from queue to displayed, re-render. */
    _twTick() {
        const tw = this._tw;
        tw.timer = null;

        if (tw.queue.length === 0) {
            // Nothing left; if SSE is done, fire finish callback
            if (tw.done && tw.onFinish) {
                tw.onFinish();
                tw.onFinish = null;
            }
            return;
        }

        // Adaptive speed: faster when queue is large, slower when small
        const pending = tw.queue.length;
        let chars = tw.charsPerFrame;
        if (pending > 200) chars = Math.min(20, Math.ceil(pending / 10));
        else if (pending > 50) chars = Math.min(10, Math.ceil(pending / 5));

        const chunk = tw.queue.slice(0, chars);
        tw.queue = tw.queue.slice(chars);
        tw.displayed += chunk;

        this.renderMarkdown(tw.bubbleEl, tw.displayed);
        this.scrollToBottom();

        // Continue animation
        this._twSchedule();
    },

    /** Flush the entire remaining queue immediately (e.g., on error or finish). */
    _twFlush() {
        const tw = this._tw;
        if (tw.timer) {
            cancelAnimationFrame(tw.timer);
            tw.timer = null;
        }
        if (tw.queue.length > 0) {
            tw.displayed += tw.queue;
            tw.queue = '';
            this.renderMarkdown(tw.bubbleEl, tw.displayed);
            this.scrollToBottom();
        }
    },

    /** Send a user message and start SSE streaming. */
    async sendMessage(message) {
        if (!message.trim() || this.isStreaming) return;
        if (!SessionManager.currentSessionId) {
            await SessionManager.createSession();
        }

        // Display user message
        this.appendMessage('user', message);

        // Force scroll to bottom when the user submits a new message.
        // This resets any previous scroll-up state so the user sees the
        // new assistant response from the start.
        this.forceScrollToBottom();

        // Disable input during streaming, show Stop button
        this.isStreaming = true;
        this._abortController = new AbortController();
        document.getElementById('btn-send').classList.add('hidden');
        document.getElementById('btn-stop').classList.remove('hidden');
        document.getElementById('user-input').value = '';
        // Reset textarea height
        document.getElementById('user-input').style.height = 'auto';

        // Create assistant bubble for streaming and show a placeholder
        // so the user sees immediate feedback instead of an empty bubble.
        const bubbleEl = this.createAssistantBubble();
        bubbleEl.innerHTML = '<span class="working-indicator">Working...</span>';
        let fullContent = '';
        let firstTokenReceived = false;

        // Scroll again after the assistant bubble is added so both the
        // user message and the "Working..." indicator are visible.
        this.forceScrollToBottom();

        // Initialize typewriter for this bubble
        this._twStart(bubbleEl);

        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: SessionManager.currentSessionId,
                    message: message,
                }),
                signal: this._abortController.signal,
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({ detail: resp.statusText }));
                throw new Error(errData.detail || `HTTP ${resp.status}`);
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            let streamDone = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // SSE events are separated by double newlines.
                let boundary;
                while ((boundary = buffer.indexOf('\n\n')) !== -1) {
                    const eventBlock = buffer.slice(0, boundary);
                    buffer = buffer.slice(boundary + 2);

                    for (const line of eventBlock.split('\n')) {
                        if (!line.startsWith('data:')) continue;
                        const dataStr = line.slice(5).trim();
                        if (!dataStr) continue;
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.content) {
                                // Clear the "Working..." placeholder on the
                                // first real token so the typewriter takes over.
                                if (!firstTokenReceived) {
                                    firstTokenReceived = true;
                                    bubbleEl.innerHTML = '';
                                    // Add the copy button now that the placeholder
                                    // is gone and won't destroy it.
                                    this._addCopyButton(bubbleEl);
                                }
                                fullContent += data.content;
                                // Push to typewriter queue instead of rendering directly
                                this._twPush(data.content);
                            }
                            // -- Reasoning (thinking) tokens ----------------
                            // Reasoning tokens are sent as {"reasoning": "..."}
                            // by the backend when the LLM produces thinking
                            // output (e.g. Ollama qwen3, DeepSeek-R1).
                            // We display them in a collapsible panel above the
                            // main content bubble so users can optionally
                            // inspect the model's chain-of-thought.
                            if (data.reasoning) {
                                // Clear the "Working..." placeholder when the
                                // first reasoning token arrives.
                                if (!firstTokenReceived) {
                                    firstTokenReceived = true;
                                    bubbleEl.innerHTML = '';
                                    this._addCopyButton(bubbleEl);
                                }
                                this._reasoning.text += data.reasoning;
                                // Show the panel on first reasoning token
                                if (this._reasoning.panelEl) {
                                    this._reasoning.panelEl.classList.remove('hidden');
                                    // Auto-open the panel while streaming so
                                    // the user can watch the thought process
                                    this._reasoning.panelEl.open = true;
                                }
                                // Render reasoning as Markdown for rich display
                                if (this._reasoning.contentEl) {
                                    this.renderMarkdown(
                                        this._reasoning.contentEl,
                                        this._reasoning.text
                                    );
                                }
                                this.scrollToBottom();
                            }
                            if (data.error) {
                                fullContent += `\n\n**Error:** ${data.error}`;
                                this._twPush(`\n\n**Error:** ${data.error}`);
                            }
                            // The backend sends {"status": "complete"} when
                            // the agent has finished and all data has been
                            // persisted.  Use this as the authoritative signal
                            // to exit the reader loop instead of waiting for
                            // the TCP connection to close, which may be delayed
                            // by proxies, keep-alive, or server-side cleanup.
                            if (data.status === 'complete' || data.error) {
                                streamDone = true;
                            }
                        } catch (e) {
                            // Not valid JSON, skip
                        }
                    }
                }

                // Exit the reader loop immediately once the backend signals
                // completion.  Any remaining bytes in the buffer are irrelevant
                // after the "complete" event.
                if (streamDone) break;
            }

            // SSE stream ended — wait for typewriter to finish draining
            this._tw.done = true;
            if (this._tw.queue.length > 0) {
                await new Promise(resolve => {
                    this._tw.onFinish = resolve;
                    this._twSchedule();
                });
            }

            // Collapse the reasoning panel now that streaming is complete.
            // The user can still expand it manually to review the thinking.
            // Also update the summary text to indicate completion.
            if (this._reasoning.panelEl && this._reasoning.text) {
                this._reasoning.panelEl.open = false;
                const summary = this._reasoning.panelEl.querySelector('summary');
                if (summary) {
                    summary.textContent = 'Thinking (click to expand)';
                }
            }

            // If no content was streamed, show a fallback
            if (!fullContent) {
                fullContent = 'Processing complete. Please check the results.';
                this.renderMarkdown(bubbleEl, fullContent);
            }

            // Check for report references
            ReportModule.checkForReports(fullContent);

            // Refresh session list to pick up auto-generated title
            await SessionManager.loadSessions();

        } catch (err) {
            this._twFlush();
            // Don't show error for user-initiated abort
            if (err.name !== 'AbortError') {
                fullContent = `**Error:** ${err.message}`;
                this.renderMarkdown(bubbleEl, fullContent);
            }
        } finally {
            this.isStreaming = false;
            this._abortController = null;
            document.getElementById('btn-send').classList.remove('hidden');
            document.getElementById('btn-stop').classList.add('hidden');
            // Reset scroll tracking when streaming ends so the next
            // scroll-to-bottom call works unconditionally.
            this._userScrolledUp = false;
            this.scrollToBottom();
        }
    },

    /** Append a rendered message to the chat area. */
    appendMessage(role, content) {
        const container = document.getElementById('messages');
        const msgEl = document.createElement('div');
        msgEl.className = `message ${role}`;

        const label = document.createElement('div');
        label.className = 'role-label';
        label.textContent = role === 'user' ? 'You' : 'Assistant';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        this.renderMarkdown(bubble, content);
        this._addCopyButton(bubble);

        msgEl.appendChild(label);
        msgEl.appendChild(bubble);
        container.appendChild(msgEl);
        this.scrollToBottom();
    },

    /** Create an empty assistant bubble for streaming into.
     *
     * Also prepares a hidden reasoning (thinking) panel above the bubble.
     * The panel is a <details>/<summary> element that starts hidden and
     * only becomes visible when the first reasoning token arrives.  This
     * avoids showing an empty collapsible for models that don't produce
     * reasoning output.
     */
    createAssistantBubble() {
        const container = document.getElementById('messages');
        const msgEl = document.createElement('div');
        msgEl.className = 'message assistant';

        const label = document.createElement('div');
        label.className = 'role-label';
        label.textContent = 'Assistant';

        // -- Reasoning (thinking) collapsible panel ---------------------
        // Created eagerly but hidden by default.  Shown on first reasoning
        // token in the SSE event handler.
        const reasoningPanel = document.createElement('details');
        reasoningPanel.className = 'reasoning-panel hidden';

        const reasoningSummary = document.createElement('summary');
        reasoningSummary.textContent = 'Thinking...';
        reasoningPanel.appendChild(reasoningSummary);

        const reasoningContent = document.createElement('div');
        reasoningContent.className = 'reasoning-content';
        reasoningPanel.appendChild(reasoningContent);

        // Store references for the streaming handler to append tokens
        this._reasoning.panelEl = reasoningPanel;
        this._reasoning.contentEl = reasoningContent;
        this._reasoning.text = '';

        // -- Main content bubble ----------------------------------------
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        // NOTE: copy button is NOT added here for streaming bubbles.
        // The "Working..." placeholder set in sendMessage() would destroy
        // it via innerHTML assignment.  Instead, the copy button is added
        // when the first content token arrives (see sendMessage).

        msgEl.appendChild(label);
        msgEl.appendChild(reasoningPanel);
        msgEl.appendChild(bubble);
        container.appendChild(msgEl);
        return bubble;
    },

    /** Render Markdown content into a DOM element.
     *  After rendering, injects copy buttons into each <pre> code block.
     *  Preserves any existing bubble-level copy button (.bubble-copy-btn)
     *  by detaching it before innerHTML replacement and re-appending after. */
    renderMarkdown(el, content) {
        if (typeof marked !== 'undefined') {
            // Detach the bubble copy button before wiping innerHTML so it
            // survives the re-render.  The typewriter calls this method on
            // every animation tick, so without this the button would be
            // destroyed and recreated constantly.
            const existingCopyBtn = el.querySelector(':scope > .bubble-copy-btn');
            if (existingCopyBtn) existingCopyBtn.remove();

            el.innerHTML = marked.parse(content);

            // Re-append the preserved copy button
            if (existingCopyBtn) el.appendChild(existingCopyBtn);

            // Highlight code blocks
            if (typeof hljs !== 'undefined') {
                el.querySelectorAll('pre code').forEach(block => {
                    hljs.highlightElement(block);
                });
            }
            // Add copy button to each code block
            el.querySelectorAll('pre').forEach(pre => {
                // Avoid duplicating buttons on re-render (typewriter calls
                // renderMarkdown repeatedly as new tokens arrive).
                if (pre.querySelector('.code-copy-btn')) return;
                const btn = document.createElement('button');
                btn.className = 'code-copy-btn';
                btn.title = 'Copy code';
                btn.innerHTML = this._copyIcon();
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const code = pre.querySelector('code');
                    const text = code ? code.textContent : pre.textContent;
                    this._copyToClipboard(text, btn);
                });
                pre.appendChild(btn);
            });
        } else {
            el.textContent = content;
        }
    },

    /**
     * SVG icon for the copy button (clipboard outline).
     * Returns an inline SVG string — no external icon library needed.
     */
    _copyIcon() {
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
    },

    /** SVG icon for the "copied" confirmation (checkmark). */
    _checkIcon() {
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    },

    /**
     * Copy text to clipboard and show brief checkmark feedback on the button.
     * Falls back to the legacy execCommand API for older browsers.
     */
    _copyToClipboard(text, btn) {
        const originalHTML = btn.innerHTML;
        const showSuccess = () => {
            btn.innerHTML = this._checkIcon();
            btn.classList.add('copied');
            setTimeout(() => {
                btn.innerHTML = originalHTML;
                btn.classList.remove('copied');
            }, 1500);
        };

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(showSuccess).catch(() => {
                // Fallback for clipboard API failure
                this._fallbackCopy(text);
                showSuccess();
            });
        } else {
            this._fallbackCopy(text);
            showSuccess();
        }
    },

    /** Fallback copy using a temporary textarea and execCommand. */
    _fallbackCopy(text) {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    },

    /**
     * Add a copy button to a bubble element.
     * The button copies the bubble's plain-text content to the clipboard.
     */
    _addCopyButton(bubble) {
        const btn = document.createElement('button');
        btn.className = 'bubble-copy-btn';
        btn.title = 'Copy message';
        btn.innerHTML = this._copyIcon();
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            // Clone the bubble, remove all copy buttons from the clone,
            // then extract the plain text.  This avoids including any
            // button artifacts in the copied text.
            const clone = bubble.cloneNode(true);
            clone.querySelectorAll('.bubble-copy-btn, .code-copy-btn').forEach(b => b.remove());
            const text = clone.innerText.trim();
            this._copyToClipboard(text, btn);
        });
        bubble.appendChild(btn);
    },

    /** Abort the current streaming request. */
    stopStreaming() {
        if (this._abortController) {
            this._abortController.abort();
        }
        // Flush any remaining queued text immediately
        this._twFlush();
    },

    /** Clear all messages from the chat area. */
    clearMessages() {
        document.getElementById('messages').innerHTML = '';
    },

    /**
     * Conditionally scroll the message area to the bottom.
     *
     * If the user has manually scrolled up (to review earlier content),
     * this method is a no-op — it preserves the user's reading position.
     * Auto-scroll resumes only when the user scrolls back to the bottom.
     *
     * Call forceScrollToBottom() to bypass this check (e.g. when the user
     * submits a new message).
     */
    scrollToBottom() {
        if (this._userScrolledUp) return;
        const container = document.getElementById('messages');
        container.scrollTop = container.scrollHeight;
    },

    /**
     * Unconditionally scroll to the bottom and reset the scroll-up flag.
     *
     * Used when the user submits a new message — they expect to see the
     * latest content regardless of their previous scroll position.
     */
    forceScrollToBottom() {
        this._userScrolledUp = false;
        const container = document.getElementById('messages');
        container.scrollTop = container.scrollHeight;
    },

    /**
     * Attach a scroll event listener to the messages container.
     *
     * Called once during app initialization.  The listener detects whether
     * the user has scrolled away from the bottom and sets/clears the
     * _userScrolledUp flag accordingly.
     *
     * A tolerance of 30px accounts for sub-pixel rounding and minor
     * scroll jitter that can occur during rapid content insertion.
     */
    initScrollTracking() {
        const container = document.getElementById('messages');
        container.addEventListener('scroll', () => {
            // Distance from the current scroll position to the very bottom.
            // scrollHeight = total content height
            // scrollTop    = pixels scrolled from top
            // clientHeight = visible viewport height
            const distanceFromBottom =
                container.scrollHeight - container.scrollTop - container.clientHeight;

            // Consider "at bottom" if within 30px tolerance.
            this._userScrolledUp = distanceFromBottom > 30;
        });
    },
};
