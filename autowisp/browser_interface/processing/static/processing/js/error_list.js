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
