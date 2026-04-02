/**
 * Main app logic — initializes modules, binds events, and coordinates routing.
 */

document.addEventListener('DOMContentLoaded', async () => {
    // Initialize: load sessions and create a default one if none exists
    await SessionManager.loadSessions();
    if (!SessionManager.currentSessionId) {
        await SessionManager.createSession();
    }

    // Initialize smart auto-scroll tracking on the messages container.
    // This must be called once before any streaming begins.
    ChatModule.initScrollTracking();

    // Bind "New Chat" button
    document.getElementById('btn-new-session').addEventListener('click', () => {
        SessionManager.createSession();
    });

    // Bind send button
    document.getElementById('btn-send').addEventListener('click', () => {
        const input = document.getElementById('user-input');
        ChatModule.sendMessage(input.value);
    });

    // Bind stop button
    document.getElementById('btn-stop').addEventListener('click', () => {
        ChatModule.stopStreaming();
    });

    const userInput = document.getElementById('user-input');
    const ctrlEnterCheckbox = document.getElementById('chk-ctrl-enter');

    // Track CJK IME composition state.
    // During composition (e.g. Chinese pinyin input), we must not
    // intercept Enter or aggressively resize, as this breaks IME
    // backspace and character selection.
    let isComposing = false;
    userInput.addEventListener('compositionstart', () => { isComposing = true; });
    userInput.addEventListener('compositionend', () => {
        isComposing = false;
        // Trigger resize after composition ends
        autoResize(userInput);
    });

    // Update placeholder text when checkbox state changes
    ctrlEnterCheckbox.addEventListener('change', () => {
        userInput.placeholder = ctrlEnterCheckbox.checked
            ? 'Ask about GitHub projects... (Ctrl+Enter to send)'
            : 'Ask about GitHub projects... (Enter to send)';
    });

    // Key binding: Enter sends by default; when checkbox is checked, Ctrl+Enter sends.
    // Check both isComposing flag AND e.isComposing for full CJK IME support.
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.isComposing && !isComposing) {
            if (ctrlEnterCheckbox.checked) {
                // Ctrl+Enter mode: only send on Ctrl+Enter
                if (e.ctrlKey) {
                    e.preventDefault();
                    document.getElementById('btn-send').click();
                }
            } else {
                // Default mode: Enter sends, Shift+Enter for newline
                if (!e.shiftKey) {
                    e.preventDefault();
                    document.getElementById('btn-send').click();
                }
            }
        }
    });

    // Auto-resize textarea — skip during IME composition to avoid
    // breaking CJK backspace behavior.
    // Max height ~420px allows up to 20 visible lines before scrollbar appears.
    function autoResize(el) {
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 420) + 'px';
    }

    userInput.addEventListener('input', function () {
        if (!isComposing) {
            autoResize(this);
        }
    });

    // Report panel buttons
    document.getElementById('btn-close-report').addEventListener('click', () => {
        ReportModule.closePanel();
    });
    document.getElementById('btn-download-report').addEventListener('click', () => {
        ReportModule.downloadReport();
    });
});
