//Shift the displayed sub-processes by the specified amount (positive or
//negative).
function scroll_subprocesses(shift)
{
    let step = Math.sign(shift);
    if( shift < 0 ) {
        let sub_id;
        for( sub_id = scroll_subprocesses.first_visible; 
             sub_id >= Math.max(0, scroll_subprocesses.first_visible + shift);
             sub_id -= 1 ) {
            document.getElementById("Sub" + sub_id).style.display = "inline-block";
        }
        scroll_subprocesses.first_visible = sub_id + 1;
    } else {
        let sub_id;
        for( sub_id = scroll_subprocesses.first_visible; 
             sub_id < Math.min(scroll_subprocesses.num_subprocesses - 1, 
                               scroll_subprocesses.first_visible + shift);
             sub_id += 1 ) {
            document.getElementById("Sub" + sub_id).style.display = "none";
        }
        scroll_subprocesses.first_visible = sub_id;
    }
    decorate();
}

function decorate() 
{
    if (scroll_subprocesses.first_visible > 0) {
        document.getElementById("scroll-up").innerHTML = "&#x25b2";
    } else {
        document.getElementById("scroll-up").innerHTML = "";
    }
    if (scroll_subprocesses.first_visible < scroll_subprocesses.num_subprocesses - 1) {
        document.getElementById("scroll-down").innerHTML = "&#x25bc";
    } else {
        document.getElementById("scroll-down").innerHTML = "";
    }
}

(
    function()
    {
        scroll_subprocesses.first_visible = 0;
        scroll_subprocesses.num_subprocesses = 
            document.getElementsByClassName("subp-select").length;
        document.getElementById("scroll-up").onclick = 
            (function(){scroll_subprocesses(-1);});
        document.getElementById("scroll-down").onclick = 
            (function() {scroll_subprocesses(1);});
        decorate();
    }
)()

