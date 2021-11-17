$(document).ready(function() {
	$('#fullpage').fullpage({
		autoScrolling: true,
		scrollHorizontally: true,
		navigation: true,
		navigationPosition: 'right',
		navigationTooltips: ['자문', '소개', '문의'],
		showActiveTooltip: true,
		slidesNavigation: true,
		slidesNavPosition: 'bottom',
		fixedElements: '.switch',

		//디자인
		sectionsColor: ['#905bb5', '#fafafa', '#fafafa'],
	});
});

jQuery(document).ready(function(){
   $('.title').mousemove(function(e){
     var rXP = (e.pageX - this.offsetLeft-$(this).width()/2);
     var rYP = (e.pageY - this.offsetTop-$(this).height()/2);
     $('.title h1').css('text-shadow', +rYP/10+'px '+rXP/80+'px #905bb5, '+rYP/8+'px '+rXP/60+'px #feb737');
   });
});
