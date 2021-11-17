$(function() {
    $(window).scroll(function() {
        var winTop = $(window).scrollTop();
        if(winTop >= 10){
            $("body").addClass("sticky-header");
        } else {
            $("body").removeClass("sticky-header");
        }
    });

    $("textarea").on('keydown keyup', function() {
        $(this).height(1).height( $(this).prop('scrollHeight') + 12 );
    });
});
