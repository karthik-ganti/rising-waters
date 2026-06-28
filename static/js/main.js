/* Rising Waters — main.js */

document.addEventListener('DOMContentLoaded', function () {
    animateCounters();
    initFormValidation();
    initPredictButton();
});

/* ── Animated stat counters on home page ─────────────────── */
function animateCounters() {
    const counters = document.querySelectorAll('.rw-stat-number[data-target]');
    if (!counters.length) return;

    const observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const target = parseFloat(el.getAttribute('data-target'));
            const isDecimal = target % 1 !== 0;
            const duration = 1200;
            const step = 16;
            const increment = target / (duration / step);
            let current = 0;

            const timer = setInterval(function () {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                el.textContent = isDecimal ? current.toFixed(2) : Math.floor(current);
            }, step);

            observer.unobserve(el);
        });
    }, { threshold: 0.4 });

    counters.forEach(function (el) { observer.observe(el); });
}

/* ── Client-side prediction form validation ──────────────── */
function initFormValidation() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    const rules = {
        annual_rainfall:  { min: 0,   max: 10000, label: 'Annual Rainfall' },
        cloud_visibility: { min: 0,   max: 100,   label: 'Cloud Visibility' },
        temperature:      { min: -10, max: 60,    label: 'Temperature' },
        humidity:         { min: 0,   max: 100,   label: 'Humidity' },
        seasonal_rainfall:{ min: 0,   max: 8000,  label: 'Seasonal Rainfall' }
    };

    form.addEventListener('submit', function (e) {
        let valid = true;

        Object.entries(rules).forEach(function ([name, rule]) {
            const input = form.querySelector('[name="' + name + '"]');
            if (!input) return;
            clearError(input);

            const val = parseFloat(input.value);
            if (isNaN(val)) {
                showError(input, rule.label + ' must be a number.');
                valid = false;
            } else if (val < rule.min || val > rule.max) {
                showError(input, rule.label + ' must be between ' + rule.min + ' and ' + rule.max + '.');
                valid = false;
            }
        });

        if (!valid) e.preventDefault();
    });

    function showError(input, msg) {
        input.classList.add('is-invalid');
        let feedback = input.parentElement.querySelector('.invalid-feedback');
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            input.parentElement.appendChild(feedback);
        }
        feedback.textContent = msg;
    }

    function clearError(input) {
        input.classList.remove('is-invalid');
        const feedback = input.parentElement.querySelector('.invalid-feedback');
        if (feedback) feedback.textContent = '';
    }
}

/* ── Predict button loading state ────────────────────────── */
function initPredictButton() {
    const form = document.getElementById('predictionForm');
    const btn  = document.getElementById('predictBtn');
    if (!form || !btn) return;

    form.addEventListener('submit', function () {
        if (form.checkValidity()) {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analysing...';
        }
    });
}
