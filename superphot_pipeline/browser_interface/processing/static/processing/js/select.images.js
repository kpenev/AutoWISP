function swapFilterType(which)
{
    let button = document.getElementById(which + 'filter_type_button');
    let result = document.getElementById(which + 'filter_type');
    if ( button.value === 'Regular Expression' ) {
        button.value = 'Shell Pattern';
        result.value = 'Shell Pattern';
    } else {
        button.value = 'Regular Expression';
        result.value = 'Regular Expression';
    }
    document.getElementById('update_img_selector').submit();
}

function enterDirectory(dirname)
{
    form = document.getElementById('update_img_selector');

    input = document.createElement('input');
    input.setAttribute('name', 'enter_dir');
    input.setAttribute('value', dirname);
    input.setAttribute('type', 'hidden');

    form.append(input);

    form.submit();
}
