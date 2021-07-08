function getBathValue() {
  var uiBathrooms = document.getElementsByName("uiBathrooms");
  for(var i in uiBathrooms) {
    if(uiBathrooms[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function getBHKValue() {
  var uiBHK = document.getElementsByName("uiBHK");
  for(var i in uiBHK) {
    if(uiBHK[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}


function getNRoomValue() {
    var uiNrooms = document.getElementsByName("uiNrooms");
    for (var i in uiNrooms) {
        if (uiNrooms[i].checked) {
            return parseInt(i) + 1;
        }
    }
    return -1; // Invalid Value
}

function getDISTValue() {
    var uiDistMainRoad = document.getElementsByName("uiDistMainRoad");
    for (var i in uiDistMainRoad) {
        if (uiDistMainRoad[i].checked) {
            return parseInt(i) + 1;
        }
    }
    return -1; // Invalid Value
}

function onClickedEstimatePrice() {
  console.log("Estimate price button clicked");
  var INT_SQFT = document.getElementById("uiSqft");
  var DIST_MAINROAD = getNRoomValue();
  var N_BEDROOM = getBHKValue();
  var N_BATHROOM = getBathValue();
  var N_ROOM = getNRoomValue();
  var location = document.getElementById("uiLocations");
  var estPrice = document.getElementById("uiEstimatedPrice");

  var url = "http://127.0.0.1:5000/predict_home_price"; //Use this if you are NOT using nginx which is first 7 tutorials
  // var url = "/api/predict_home_price"; // Use this if  you are using nginx. i.e tutorial 8 and onwards

  $.post(url, {
    INT_SQFT: parseFloat(INT_SQFT.value),
    DIST_MAINROAD: DIST_MAINROAD,
    N_BEDROOM: N_BEDROOM,
    N_BATHROOM: N_BATHROOM,
    N_ROOM: N_ROOM,
    location: location.value,
  },function(data, status) {
        console.log(data.estimated_price);
        estPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " Lakh</h2>";
        console.log(status);
  });
}

function onPageLoad() {
  console.log( "document loaded" );
  var url = "http://127.0.0.1:5000/get_location_names"; // Use this if you are NOT using nginx which is first 7 tutorials
  // var url = "/api/get_location_names"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
  $.get(url,function(data, status) {
      console.log("got response for get_location_names request");
      if(data) {
          var locations = data.locations;
          var uiLocations = document.getElementById("uiLocations");
          $(uiLocations).empty();
          for(var i in locations) {
              var opt = new Option(locations[i]);
              $(uiLocations).append(opt);
          }
      }
  });
}

window.onload = onPageLoad;
