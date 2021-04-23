var types = ["eyes_dorsal", "heart", "outline_dorsal", "outline_lateral", "ov", "yolk"];

  document.getElementById("comp_select").onchange = function(){
  var path = document.getElementById("comp_select").value;
  var div = document.getElementById("lateral_div");
  div.innerHTML = '<img src={{url_for(\'static\',filename= \'\')}}'+ path.substring(7) +'/lateral.png' + '/>';
  var div2 = document.getElementById("dorsal_div");
  div2.innerHTML = '<img src={{url_for(\'static\',filename= \'\')}}'+ path.substring(7) +'/dorsal.png' + '/>';
  types.forEach((item) => {
    calc(item);
  });

 };

function calc(type)
  {
  var e = document.getElementById("comp_select");
  var path = e.value;
  if (document.getElementById(type).checked)
  {
    var image = path + "/" + type + "_out.png"
    var div = document.getElementById(type+"_div");
    div.innerHTML = '<img src={{url_for(\'static\',filename= \'\')}}'+image.substring(7)+ '/>';
  }
  else{
    document.getElementById(type+"_div").innerHTML = '';
  }
};
