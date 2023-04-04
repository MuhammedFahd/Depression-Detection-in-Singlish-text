/**
* File Description: Contains functions related to depression detector
* Author: Fahd Fazal
*/

$(document).ready(function() {
    $("#detectBtn").click(function() {
        $("#inputTxt").LoadingOverlay("show");
        $("#depressionStatus").html("");
        $("#depressionRate").html("");
        $("#alertBox").empty();
        $("#resultsBtn").prop('disabled', true);
      
        var txt = $("#inputTxt").val();
        if (txt != '') {
            $.ajax({
                type: "POST",
                url: azure_api_url + "/detect",
                data: JSON.stringify({text: txt}),
                headers: {
                'Access-Control-Allow-Origin': '*',
                },
                crossDomain:true,
                contentType: 'application/json',
                success: function (response) {
                    $("#inputTxt").LoadingOverlay("hide", true);

                    var successAlert = '<div class="alert alert-success alert-dismissible fade show" role="alert">' +
                        '<span><i class="bi bi-info-circle-fill me-2"></i>Successfully detected! click view results button for final result</span>' +
                        '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
                    $("#alertBox").html(successAlert);

                    if (response.prediction == 1) {
                        $("#depressionStatus").html("The person who posted this text can be suffering from depression!");
                        $("#depressionRate").html("Estimated depression rate: " + (response.percentage * 100) + "%");
                    } else {
                        $("#depressionStatus").html("The person who posted this text is not suffering from depression!");
                    }

                    $("#resultsBtn").prop('disabled', false);
                },
                error: function (xhr, status, error) {
                    $("#inputTxt").LoadingOverlay("hide", true);
                    var errorAlert = '<div class="alert alert-danger alert-dismissible fade show" role="alert">' +
                        '<span><i class="bi bi-x-octagon-fill me-2"></i>Something went wrong!</span>' +
                        '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
                    $("#alertBox").html(errorAlert);
                }
            });
        } else {
            var warningAlert = '<div class="alert alert-warning alert-dismissible fade show" role="alert">' +
                '<span><i class="bi bi-exclamation-triangle-fill me-2"></i>Textbox cannot be empty! Please enter a Singlish text</span>' +
                '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
            $("#alertBox").html(warningAlert);
            $("#inputTxt").LoadingOverlay("hide", true);
        }
    });
    
    $("#resultsBtn").click(function() {
        $("#alertBox").empty();
        $("#msgModal").modal('show');
    });
});