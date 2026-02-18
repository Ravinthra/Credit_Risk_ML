/* ============================================================
   Credit Risk AI — UI Interactions & Animations
   ============================================================ */

document.addEventListener('DOMContentLoaded', function () {

    // ─── Scroll-triggered Animations ─────────────────────
    const animElements = document.querySelectorAll('.animate-in');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.05 });

    animElements.forEach(el => {
        el.style.animationPlayState = 'paused';
        observer.observe(el);
    });

    // ─── Form Focus Effects ──────────────────────────────
    const inputs = document.querySelectorAll('.form-input, .form-select');

    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.closest('.form-group')?.classList.add('focused');
        });

        input.addEventListener('blur', () => {
            input.closest('.form-group')?.classList.remove('focused');
            if (input.value && input.value !== input.defaultValue) {
                input.classList.add('has-value');
            }
        });

        // Initialize has-value on load
        if (input.value && input.value !== '') {
            input.classList.add('has-value');
        }
    });

    // ─── Animated Gauge Counter ──────────────────────────
    const gaugePercentage = document.querySelector('.gauge-percentage');
    if (gaugePercentage) {
        const target = parseFloat(gaugePercentage.dataset.target) || 0;
        animateCounter(gaugePercentage, 0, target, 1200);
    }

    // Gauge fill animation
    const gaugeFill = document.getElementById('gauge-fill');
    if (gaugeFill) {
        const color = gaugeFill.dataset.color;
        const value = parseFloat(gaugeFill.dataset.value) || 0;
        requestAnimationFrame(() => {
            gaugeFill.style.cssText = `
                position: absolute; top: 0; left: 0;
                width: 200px; height: 200px; border-radius: 50%;
                background: conic-gradient(from 180deg at 50% 50%, ${color} ${value}turn, transparent 0);
                transform: rotate(270deg);
                clip-path: polygon(0 50%, 100% 50%, 100% 0, 0 0);
                transition: all 1s cubic-bezier(0.4, 0, 0.2, 1);
            `;
        });
    }

    // ─── Apply data-color attributes ─────────────────────
    // Moves Django template colors out of style="" to avoid CSS linter errors
    document.querySelectorAll('[data-color]').forEach(el => {
        const color = el.dataset.color;
        if (!color) return;

        // Gauge percentage & percent sign
        if (el.classList.contains('gauge-percentage') || el.classList.contains('gauge-percent-sign')) {
            el.style.color = color;
            if (el.classList.contains('gauge-percent-sign')) {
                el.style.fontSize = '1.2rem';
            }
        }

        // Risk badge: tinted background + border
        if (el.classList.contains('risk-badge')) {
            el.style.background = color + '15';
            el.style.color = color;
            el.style.border = '1.5px solid ' + color + '40';
        }

        // Action box: tinted background + left border
        if (el.classList.contains('action-box')) {
            el.style.background = color + '08';
            el.style.borderLeft = '3px solid ' + color;
        }
    });

    // ─── SHAP Bar Animation ──────────────────────────────
    const shapBars = document.querySelectorAll('.shap-bar-fill[data-width]');
    const shapObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bar = entry.target;
                const width = bar.dataset.width;
                // Start from 0 and animate to target width
                requestAnimationFrame(() => {
                    bar.style.width = Math.min(parseFloat(width), 100) + '%';
                });
                shapObserver.unobserve(bar);
            }
        });
    }, { threshold: 0.1 });

    shapBars.forEach(bar => {
        bar.style.width = '0%'; // Start collapsed
        shapObserver.observe(bar);
    });

    // ─── Feature Bar Animation (Dashboard) ───────────────
    const featureBars = document.querySelectorAll('.dash-feat-bar[data-width]');
    const featureObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bar = entry.target;
                const width = bar.dataset.width;
                requestAnimationFrame(() => {
                    bar.style.width = Math.min(parseFloat(width), 100) + '%';
                });
                featureObserver.unobserve(bar);
            }
        });
    }, { threshold: 0.1 });

    featureBars.forEach(bar => {
        bar.style.width = '0%';
        featureObserver.observe(bar);
    });

    // ─── AUC Bar Animation (Dashboard) ───────────────────
    const aucBars = document.querySelectorAll('.dash-auc-fill[data-width]');
    const aucObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bar = entry.target;
                const width = bar.dataset.width;
                requestAnimationFrame(() => {
                    bar.style.width = Math.min(parseFloat(width), 100) + '%';
                });
                aucObserver.unobserve(bar);
            }
        });
    }, { threshold: 0.1 });

    aucBars.forEach(bar => {
        bar.style.width = '0%';
        aucObserver.observe(bar);
    });

    // ─── Submit Button Feedback ──────────────────────────
    const submitBtn = document.getElementById('submit-btn');
    const form = document.getElementById('predict-form');

    if (submitBtn && form) {
        form.addEventListener('submit', () => {
            submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
            submitBtn.disabled = true;
            submitBtn.style.opacity = '0.8';
        });
    }

    // ─── Navbar scroll effect ────────────────────────────
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        let lastScroll = 0;
        window.addEventListener('scroll', () => {
            const currentScroll = window.scrollY;
            if (currentScroll > 50) {
                navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
            } else {
                navbar.style.boxShadow = 'none';
            }
            lastScroll = currentScroll;
        }, { passive: true });
    }

});

// ─── Counter Animation Helper ────────────────────────────
function animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    const diff = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + diff * eased;

        element.textContent = current.toFixed(1);

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = end.toFixed(1);
        }
    }

    requestAnimationFrame(update);
}
