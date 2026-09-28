// Chart.js initialization for Results Page — OWASP ZAP Cybersecurity Theme
document.addEventListener("DOMContentLoaded", () => {
    if (window.severityData && document.getElementById("severityChart")) {
        const sevCtx = document.getElementById("severityChart").getContext("2d");
        new Chart(sevCtx, {
            type: "bar",
            data: {
                labels: Object.keys(window.severityData),
                datasets: [{
                    label: "Findings Count",
                    data: Object.values(window.severityData),
                    backgroundColor: [
                        "#15803d", // Low (Green)
                        "#b45309", // Medium (Amber)
                        "#c2410c", // High (Orange)
                        "#b91c1c"  // Critical (Red)
                    ],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: { beginAtZero: true, ticks: { stepSize: 1 } }
                }
            }
        });
    }

    if (window.attackData && document.getElementById("attackTypeChart")) {
        const attackCtx = document.getElementById("attackTypeChart").getContext("2d");
        new Chart(attackCtx, {
            type: "pie",
            data: {
                labels: Object.keys(window.attackData),
                datasets: [{
                    data: Object.values(window.attackData),
                    backgroundColor: [
                        "#0072ff", // ZAP Blue
                        "#00c6ff", // ZAP Electric Cyan
                        "#00a896", // ZAP Teal
                        "#028090", // ZAP Deep Teal
                        "#f59e0b", // Amber
                        "#ef4444", // Crimson
                        "#6366f1", // Indigo
                        "#8b5cf6"  // Violet
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { family: "'Plus Jakarta Sans', sans-serif", weight: '600', size: 11 }
                        }
                    }
                }
            }
        });
    }
});
