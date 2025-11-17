//checkbox height to step button height 
(function () {
  function syncHeights() {
    document.querySelectorAll('.aw-step-puzzle').forEach(function (wrap) {
      var name = wrap.querySelector('.aw-step-name');
      if (!name) return;
      var h = name.getBoundingClientRect().height;
      if (h > 0) wrap.style.setProperty('--aw-step-h', h + 'px');
    });
  }
  document.addEventListener('DOMContentLoaded', syncHeights);
  window.addEventListener('load', function () {
    syncHeights();
    setTimeout(syncHeights, 50);
  });
  window.addEventListener('resize', syncHeights);
})();

//click or keyboard toggles all step checkboxes
(function () {
  function stepBoxes() {
    return Array.from(document.querySelectorAll('.aw-step-checkbox input[name="steps"]'));
  }
  function allChecked(boxes) {
    return boxes.length > 0 && boxes.every(function (b) { return b.checked; });
  }
  function setAll(boxes, value) {
    boxes.forEach(function (b) {
      if (b.checked !== value) {
        b.checked = value;
      }
    });
  }
  function handleSelectAllActivate() {
    var selectAllBtn = document.getElementById('aw-select-all');
    if (selectAllBtn && selectAllBtn.classList.contains('disabled')) {
      return; //Dont do anything if disabled
    }
    
    var boxes = stepBoxes();
    if (boxes.length === 0) return;
    var targetState = !allChecked(boxes);
    setAll(boxes, targetState);
  }

  document.addEventListener('click', function (e) {
    var btn = e.target && e.target.closest('#aw-select-all');
    if (btn) handleSelectAllActivate();
  });

  document.addEventListener('keydown', function (e) {
    var focused = document.activeElement;
    if (focused && focused.id === 'aw-select-all' && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      handleSelectAllActivate();
    }
  });

  //each checkbox is now independent
})();

//auto-refresh
(function () {
  function setupRefresh() {
    var cfg = document.getElementById('aw-progress-config');
    if (!cfg) return;
    try { cfg.dataset.jsLoaded = '1'; } catch (e) {}
    var seconds = parseInt(cfg.dataset.refreshSeconds || '0', 10);
    var url = cfg.dataset.refreshUrl || '';
    if (seconds > 0 && url) {
      setTimeout(function () { window.location.href = url; }, seconds * 1000);
    }
  }
  document.addEventListener('DOMContentLoaded', setupRefresh);
  window.addEventListener('load', setupRefresh);
})();
