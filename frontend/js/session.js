/**
 * Session management module — handles CRUD operations for chat sessions.
 */

const SessionManager = {
    currentSessionId: null,

    /** Fetch all sessions from the API and render in sidebar. */
    async loadSessions() {
        try {
            const resp = await fetch('/api/sessions');
            const sessions = await resp.json();
            this.renderSessionList(sessions);
            // Update the header title if the current session title changed
            if (this.currentSessionId) {
                const current = sessions.find(s => s.id === this.currentSessionId);
                if (current) {
                    document.getElementById('session-title').textContent = current.title;
                }
            } else if (sessions.length > 0) {
                // On page load with no active session, select the first one
                await this.switchSession(sessions[0].id);
            }
        } catch (err) {
            console.error('Failed to load sessions:', err);
        }
    },

    /** Create a new chat session via API. Skip if an empty session already exists. */
    async createSession(title = 'New Chat') {
        try {
            // Check ALL sessions for an existing empty one (title "New Chat", no messages).
            // This prevents creating duplicates on page refresh or repeated "New Chat" clicks.
            const listResp = await fetch('/api/sessions');
            if (listResp.ok) {
                const sessions = await listResp.json();
                for (const s of sessions) {
                    if (s.title === 'New Chat') {
                        const detailResp = await fetch(`/api/sessions/${s.id}`);
                        if (detailResp.ok) {
                            const data = await detailResp.json();
                            if (data.messages.length === 0) {
                                // Found an empty session — reuse it
                                this.currentSessionId = s.id;
                                this.renderSessionList(sessions);
                                this.highlightActiveSession();
                                ChatModule.clearMessages();
                                document.getElementById('session-title').textContent = s.title;
                                return data.session;
                            }
                        }
                    }
                }
            }

            const resp = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title }),
            });
            const session = await resp.json();
            this.currentSessionId = session.id;
            await this.loadSessions();
            ChatModule.clearMessages();
            document.getElementById('session-title').textContent = session.title;
            return session;
        } catch (err) {
            console.error('Failed to create session:', err);
        }
    },

    /** Switch to an existing session and load its message history. */
    async switchSession(sessionId) {
        try {
            const resp = await fetch(`/api/sessions/${sessionId}`);
            const data = await resp.json();
            this.currentSessionId = data.session.id;
            document.getElementById('session-title').textContent = data.session.title;
            ChatModule.clearMessages();
            // Restore message history
            for (const msg of data.messages) {
                ChatModule.appendMessage(msg.role, msg.content);
            }
            this.highlightActiveSession();
        } catch (err) {
            console.error('Failed to switch session:', err);
        }
    },

    /** Delete a session with custom confirmation dialog. */
    async deleteSession(sessionId) {
        const sessionItem = document.querySelector(`.session-item[data-id="${sessionId}"] span`);
        const title = sessionItem ? sessionItem.textContent : 'this session';

        const confirmed = await this.showConfirmDialog(
            'Delete Session',
            `Delete "${title}"? All messages and reports in this session will be permanently removed.`
        );
        if (!confirmed) return;

        try {
            await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
            if (this.currentSessionId === sessionId) {
                this.currentSessionId = null;
                ChatModule.clearMessages();
                document.getElementById('session-title').textContent = 'New Chat';
            }
            await this.loadSessions();
        } catch (err) {
            console.error('Failed to delete session:', err);
        }
    },

    /** Show a custom confirmation dialog. Returns a Promise<boolean>. */
    showConfirmDialog(heading, message) {
        return new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.className = 'confirm-overlay';
            overlay.innerHTML = `
                <div class="confirm-dialog">
                    <h4>${heading}</h4>
                    <p>${message}</p>
                    <div class="btn-row">
                        <button class="btn-cancel">Cancel</button>
                        <button class="btn-confirm">Delete</button>
                    </div>
                </div>
            `;

            const close = (result) => {
                overlay.remove();
                resolve(result);
            };

            overlay.querySelector('.btn-cancel').addEventListener('click', () => close(false));
            overlay.querySelector('.btn-confirm').addEventListener('click', () => close(true));
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) close(false);
            });

            document.body.appendChild(overlay);
            overlay.querySelector('.btn-cancel').focus();
        });
    },

    /** Render the session list in the sidebar. */
    renderSessionList(sessions) {
        const container = document.getElementById('session-list');
        container.innerHTML = '';
        for (const s of sessions) {
            const el = document.createElement('div');
            el.className = 'session-item' + (s.id === this.currentSessionId ? ' active' : '');
            el.setAttribute('data-id', s.id);
            el.innerHTML = `
                <span title="${s.title}">${s.title}</span>
                <button class="delete-btn" title="Delete">&times;</button>
            `;
            // Clicking anywhere on the session item selects it
            el.addEventListener('click', () => this.switchSession(s.id));
            // Only the delete button triggers deletion; stop propagation to avoid selecting
            el.querySelector('.delete-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteSession(s.id);
            });
            container.appendChild(el);
        }
    },

    /** Highlight the active session in the sidebar. */
    highlightActiveSession() {
        document.querySelectorAll('.session-item').forEach(el => {
            el.classList.toggle('active', el.getAttribute('data-id') === this.currentSessionId);
        });
    },
};
