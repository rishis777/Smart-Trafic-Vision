/**
 * Traffic AI Command Dashboard Client
 * Handles real-time polling, Chart.js telemetry, signal animations, camera switching, and operator overrides.
 */

document.addEventListener('DOMContentLoaded', () => {
    let densityChart = null;
    let distributionChart = null;
    let isOverrideActive = false;
    let audioEnabled = true;
    let wasEmergencyActive = false;

    // Initialize Chart.js
    initCharts();

    // Start Real-Time Polling Loop (1.5 seconds)
    fetchLiveTelemetry();
    setInterval(fetchLiveTelemetry, 1500);

    // Setup Event Listeners
    setupControls();

    function initCharts() {
        const densityCtx = document.getElementById('densityChart');
        if (densityCtx) {
            densityChart = new Chart(densityCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'NORTH', data: [], borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.1)', tension: 0.35, fill: true },
                        { label: 'SOUTH', data: [], borderColor: '#34d399', backgroundColor: 'rgba(52, 211, 153, 0.1)', tension: 0.35, fill: true },
                        { label: 'EAST', data: [], borderColor: '#fbbf24', backgroundColor: 'rgba(251, 191, 36, 0.1)', tension: 0.35, fill: true },
                        { label: 'WEST', data: [], borderColor: '#f87171', backgroundColor: 'rgba(248, 113, 113, 0.1)', tension: 0.35, fill: true }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b', maxTicksLimit: 8 } },
                        y: { min: 0, max: 1.0, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b' } }
                    },
                    plugins: {
                        legend: { labels: { color: '#94a3b8', boxWidth: 12 } }
                    }
                }
            });
        }

        const distCtx = document.getElementById('distributionChart');
        if (distCtx) {
            distributionChart = new Chart(distCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Cars', 'Motorcycles', 'Buses', 'Trucks'],
                    datasets: [{
                        data: [1, 1, 1, 1],
                        backgroundColor: ['#38bdf8', '#fbbf24', '#8b5cf6', '#f43f5e'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#94a3b8', boxWidth: 10, padding: 12 } }
                    },
                    cutout: '70%'
                }
            });
        }
    }

    async function fetchLiveTelemetry() {
        try {
            const res = await fetch('/api/traffic/live');
            if (!res.ok) return;
            const data = await res.json();
            updateDashboard(data);
        } catch (err) {
            console.error("Telemetry fetch error:", err);
        }
    }

    function updateDashboard(data) {
        // 1. Stat Cards
        const totalVehEl = document.getElementById('statTotalVehicles');
        if (totalVehEl) totalVehEl.innerText = data.total_active_vehicles ?? 0;

        const avgDensEl = document.getElementById('statAvgDensity');
        if (avgDensEl) {
            const dens = data.average_density ?? 0.0;
            avgDensEl.innerText = `${Math.round(dens * 100)}%`;
        }

        const waitTimeEl = document.getElementById('statAvgWaitTime');
        if (waitTimeEl) waitTimeEl.innerText = `${data.signal?.remaining_seconds ?? 0}s`;

        const totalCountedEl = document.getElementById('statTotalCounted');
        if (totalCountedEl) totalCountedEl.innerText = data.total_counted ?? 0;

        // 2. Active Mode Badge & Countdown
        const modeBadge = document.getElementById('currentModeBadge');
        if (modeBadge) {
            modeBadge.innerText = data.signal?.mode ?? 'ADAPTIVE';
            modeBadge.className = `role-badge ${data.signal?.mode === 'MANUAL_OVERRIDE' ? 'admin' : (data.signal?.mode === 'EMERGENCY' ? 'admin' : 'control_operator')}`;
        }

        const countdownEl = document.getElementById('signalCountdown');
        if (countdownEl) countdownEl.innerText = `${data.signal?.remaining_seconds ?? 0}s`;

        const activeDirEl = document.getElementById('signalActiveDir');
        if (activeDirEl) activeDirEl.innerText = data.signal?.active_direction ?? 'NORTH';

        // Cycle Progress Bar
        const progBar = document.getElementById('cycleProgressBar');
        if (progBar && data.signal?.allocated_green_time) {
            const total = data.signal.allocated_green_time;
            const rem = data.signal.remaining_seconds ?? 0;
            const pct = Math.max(0, Math.min(100, Math.round(((total - rem) / total) * 100)));
            progBar.style.width = `${pct}%`;
        }

        // 3. Update Traffic Light Widgets
        const signals = data.signal?.signals ?? {};
        ['NORTH', 'SOUTH', 'EAST', 'WEST'].forEach(d => {
            const state = signals[d] ?? 'RED';
            const card = document.getElementById(`approachCard_${d}`);
            const redBulb = document.getElementById(`light_${d}_red`);
            const yellowBulb = document.getElementById(`light_${d}_yellow`);
            const greenBulb = document.getElementById(`light_${d}_green`);
            const metrics = data.direction_metrics?.[d] ?? {};

            if (redBulb) redBulb.className = `light-bulb red ${state === 'RED' ? 'active' : ''}`;
            if (yellowBulb) yellowBulb.className = `light-bulb yellow ${state === 'YELLOW' ? 'active' : ''}`;
            if (greenBulb) greenBulb.className = `light-bulb green ${state === 'GREEN' ? 'active' : ''}`;

            if (card) {
                if (state === 'GREEN') card.classList.add('active-green');
                else card.classList.remove('active-green');
            }

            const densEl = document.getElementById(`dens_${d}`);
            if (densEl) {
                const lvl = metrics.density_level ?? 'LOW';
                densEl.innerText = `${lvl} (${metrics.count ?? 0} veh)`;
                densEl.className = `badge-density ${lvl}`;
            }

            const countEl = document.getElementById(`count_${d}`);
            if (countEl) countEl.innerText = metrics.cumulative_counted ?? 0;
        });

        // 4. Emergency Corridor Banner & Audio Chime
        const emergencyBanner = document.getElementById('emergencyBanner');
        const isEmergency = data.emergency?.is_emergency_active;
        if (emergencyBanner) {
            if (isEmergency) {
                emergencyBanner.style.display = 'flex';
                document.getElementById('emergencyDir').innerText = data.emergency.corridor_direction;
                document.getElementById('emergencyType').innerText = data.emergency.vehicle_type?.toUpperCase();
                document.getElementById('emergencyRemaining').innerText = `${data.emergency.remaining_seconds}s`;

                if (!wasEmergencyActive) {
                    playEmergencyChime();
                    wasEmergencyActive = true;
                }
            } else {
                emergencyBanner.style.display = 'none';
                wasEmergencyActive = false;
            }
        }

        // 5. Manual Override Status Indicator
        isOverrideActive = data.signal?.is_override_active;
        const overrideNotice = document.getElementById('overrideNotice');
        const clearOverrideBtn = document.getElementById('btnClearOverride');
        if (overrideNotice && clearOverrideBtn) {
            if (isOverrideActive) {
                overrideNotice.style.display = 'block';
                document.getElementById('overrideDirectionName').innerText = data.signal.override_direction;
                clearOverrideBtn.style.display = 'inline-flex';
            } else {
                overrideNotice.style.display = 'none';
                clearOverrideBtn.style.display = 'none';
            }
        }

        // 6. Update Charts
        if (densityChart && data.direction_metrics) {
            const timeLabel = new Date().toLocaleTimeString();
            if (densityChart.data.labels.length > 12) {
                densityChart.data.labels.shift();
                densityChart.data.datasets.forEach(ds => ds.data.shift());
            }
            densityChart.data.labels.push(timeLabel);
            densityChart.data.datasets[0].data.push(data.direction_metrics.NORTH?.density ?? 0);
            densityChart.data.datasets[1].data.push(data.direction_metrics.SOUTH?.density ?? 0);
            densityChart.data.datasets[2].data.push(data.direction_metrics.EAST?.density ?? 0);
            densityChart.data.datasets[3].data.push(data.direction_metrics.WEST?.density ?? 0);
            densityChart.update('none');
        }

        if (distributionChart && data.type_totals) {
            distributionChart.data.datasets[0].data = [
                data.type_totals.car || 1,
                data.type_totals.motorcycle || 1,
                data.type_totals.bus || 1,
                data.type_totals.truck || 1
            ];
            distributionChart.update('none');
        }

        // 7. Update Active Alerts List
        renderAlerts(data.active_alerts ?? []);
    }

    function renderAlerts(alerts) {
        const container = document.getElementById('alertsContainer');
        if (!container) return;

        if (alerts.length === 0) {
            container.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem; padding: 0.5rem 0;">All intersections operating normally. No active alerts.</div>';
            return;
        }

        container.innerHTML = alerts.slice(0, 5).map(a => `
            <div style="background: rgba(255,255,255,0.03); border-left: 3px solid ${a.severity === 'CRITICAL' ? '#ef4444' : '#f59e0b'}; padding: 0.75rem; border-radius: 4px; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #fff;">${a.alert_type} - ${a.direction}</div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary);">${a.message}</div>
                </div>
                <button onclick="resolveAlert(${a.id || a.db_id})" class="btn btn-outline" style="padding: 0.25rem 0.55rem; font-size: 0.7rem;">Dismiss</button>
            </div>
        `).join('');
    }

    function playEmergencyChime() {
        if (!audioEnabled) return;
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, ctx.currentTime);
            osc.frequency.linearRampToValueAtTime(520, ctx.currentTime + 0.4);
            gain.gain.setValueAtTime(0.12, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.4);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.4);
        } catch(e) {}
    }

    function setupControls() {
        // Audio Toggle
        const audioBtn = document.getElementById('btnAudioToggle');
        if (audioBtn) {
            audioBtn.addEventListener('click', () => {
                audioEnabled = !audioEnabled;
                audioBtn.innerText = audioEnabled ? '🔊 Audio On' : '🔇 Audio Muted';
            });
        }

        // Force Override Form
        const btnOverride = document.getElementById('btnApplyOverride');
        if (btnOverride) {
            btnOverride.addEventListener('click', async () => {
                const dir = document.getElementById('overrideSelectDir').value;
                const dur = document.getElementById('overrideDuration').value || 30;
                try {
                    const res = await fetch('/api/signal/override', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ direction: dir, duration: parseInt(dur), reason: 'Operator HUD command' })
                    });
                    const d = await res.json();
                    if (!res.ok) alert(d.error || 'Failed to apply override');
                    else fetchLiveTelemetry();
                } catch (e) {
                    alert('Network error requesting override');
                }
            });
        }

        // Clear Override
        const btnClear = document.getElementById('btnClearOverride');
        if (btnClear) {
            btnClear.addEventListener('click', async () => {
                await fetch('/api/signal/clear_override', { method: 'POST' });
                fetchLiveTelemetry();
            });
        }

        // Mode Toggle
        const btnToggleMode = document.getElementById('btnToggleMode');
        if (btnToggleMode) {
            btnToggleMode.addEventListener('click', async () => {
                const target = btnToggleMode.getAttribute('data-target-mode') || 'FIXED';
                await fetch('/api/signal/mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: target })
                });
                btnToggleMode.setAttribute('data-target-mode', target === 'FIXED' ? 'ADAPTIVE' : 'FIXED');
                btnToggleMode.innerText = `Switch to ${target === 'FIXED' ? 'Adaptive' : 'Fixed-Time'}`;
                fetchLiveTelemetry();
            });
        }

        // Emergency Corridor Trigger
        const btnTriggerEmergency = document.getElementById('btnTriggerEmergency');
        if (btnTriggerEmergency) {
            btnTriggerEmergency.addEventListener('click', async () => {
                const dir = document.getElementById('emergencySelectDir').value;
                const vType = document.getElementById('emergencySelectType').value;
                await fetch('/api/emergency/trigger', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ direction: dir, vehicle_type: vType })
                });
                playEmergencyChime();
                fetchLiveTelemetry();
            });
        }

        // Emergency Corridor Clear
        const btnClearEmergency = document.getElementById('btnClearEmergency');
        if (btnClearEmergency) {
            btnClearEmergency.addEventListener('click', async () => {
                await fetch('/api/emergency/clear', { method: 'POST' });
                fetchLiveTelemetry();
            });
        }
    }

    // Expose global camera switcher
    window.switchCamera = async function(camId) {
        try {
            const res = await fetch('/api/camera/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ camera_id: camId })
            });
            if (res.ok) {
                ['cam1', 'cam2', 'cam3'].forEach(id => {
                    const btn = document.getElementById(`camBtn_${id}`);
                    if (btn) {
                        if (id === camId) {
                            btn.style.background = 'rgba(56, 189, 248, 0.15)';
                            btn.style.borderColor = 'var(--accent-cyan)';
                            btn.style.color = '#fff';
                        } else {
                            btn.style.background = 'transparent';
                            btn.style.borderColor = 'var(--border-subtle)';
                            btn.style.color = 'var(--text-secondary)';
                        }
                    }
                });
                const img = document.getElementById('liveVideoFeed');
                if (img) img.src = '/video_feed?' + new Date().getTime();
                setTimeout(fetchLiveTelemetry, 300);
            }
        } catch(e) {
            console.error('Camera switch error:', e);
        }
    };

    // Expose global alert resolver
    window.resolveAlert = async function(alertId) {
        await fetch(`/api/alerts/resolve/${alertId}`, { method: 'POST' });
        fetchLiveTelemetry();
    };
});
