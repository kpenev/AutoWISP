// Make each error row navigate to its detail page on click, except when
// the click lands on the row's action controls (resolve / delete).
(function () {
  document.addEventListener('click', function (e) {
    if (e.target.closest('.aw-error-actions')) return;
    var row = e.target.closest('tr[data-href]');
    if (row) {
      window.location.href = row.dataset.href;
    }
  });
})();

// Ticking the header's box ticks, or clears, every error shown; acting on
// the selection needs one ticked, and deleting is confirmed first.
(function () {
  function rowBoxes() {
    return document.querySelectorAll('input[name="error_ids"]');
  }

  var selectAll = document.getElementById('aw-select-all-errors');
  if (selectAll) {
    selectAll.addEventListener('change', function () {
      rowBoxes().forEach(function (box) { box.checked = selectAll.checked; });
    });
  }

  var selection = document.getElementById('error-selection');
  if (!selection) return;
  selection.addEventListener('submit', function (e) {
    var count = document.querySelectorAll(
      'input[name="error_ids"]:checked'
    ).length;
    if (count === 0) {
      e.preventDefault();
      alert('No errors selected.');
      return;
    }
    if (e.submitter && e.submitter.hasAttribute('data-confirm-delete')
        && !confirm('Delete ' + count + ' error(s) permanently?')) {
      e.preventDefault();
    }
  });
})();
