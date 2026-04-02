/**
 * Report display module — detects report generation events and shows reports.
 */

const ReportModule = {
    currentReportId: null,

    /** Check chat content for report references and show panel if found. */
    checkForReports(content) {
        // Look for report ID pattern in the message
        const match = content.match(/Report ID[:\s]*([a-f0-9-]+)/i);
        if (match) {
            this.currentReportId = match[1];
            this.showReport(this.currentReportId);
        }
    },

    /** Fetch and display a report in the side panel. */
    async showReport(reportId) {
        try {
            const resp = await fetch(`/api/reports/${reportId}`);
            const contentType = resp.headers.get('content-type');
            const panel = document.getElementById('report-panel');
            const content = document.getElementById('report-content');

            if (contentType && contentType.includes('text/html')) {
                // HTML report — render in an iframe
                const html = await resp.text();
                content.innerHTML = `<iframe srcdoc="${html.replace(/"/g, '&quot;')}" style="width:100%;height:100%;border:none;"></iframe>`;
            } else {
                // Markdown report
                const data = await resp.json();
                if (typeof marked !== 'undefined') {
                    content.innerHTML = marked.parse(data.content);
                } else {
                    content.textContent = data.content;
                }
            }

            panel.classList.remove('hidden');
        } catch (err) {
            console.error('Failed to load report:', err);
        }
    },

    /** Close the report panel. */
    closePanel() {
        document.getElementById('report-panel').classList.add('hidden');
    },

    /** Download the current report. */
    async downloadReport() {
        if (!this.currentReportId) return;
        window.open(`/api/reports/${this.currentReportId}/download`, '_blank');
    },

    /** Load reports for the current session (for the panel). */
    async loadSessionReports(sessionId) {
        try {
            const resp = await fetch(`/api/sessions/${sessionId}/reports`);
            return await resp.json();
        } catch (err) {
            console.error('Failed to load reports:', err);
            return [];
        }
    },
};
