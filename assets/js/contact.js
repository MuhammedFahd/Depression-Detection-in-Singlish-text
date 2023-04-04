/**
* File Description: Contains functions related to contact option
* Author: Fahd Fazal
*/

function convertFormToJSON(form) {
    return $(form)
      .serializeArray()
      .reduce(function (json, { name, value }) {
        json[name] = value;
        return json;
    }, {});
}


$(document).ready(function() {
    $("#contactForm").submit(function(e) {
        e.preventDefault();
        $(this).LoadingOverlay("show"); 
        $("#alertBox").empty();
        var form = $(this);
        var formData = convertFormToJSON(form);

        $.ajax({
            type: "POST",
            url: azure_api_url + "/contact",
            data: JSON.stringify(formData),
            headers: {
            'Access-Control-Allow-Origin': '*',
            },
            crossDomain:true,
            contentType: 'application/json',
            success: function (response) {
                $("#contactForm").LoadingOverlay("hide", true);

                var successAlert = '<div class="alert alert-success alert-dismissible fade show" role="alert">' +
                '<span><i class="bi bi-info-circle-fill me-2"></i>Message sent successfully!</span>' +
                '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
                $("#alertBox").html(successAlert);
            },
            error: function (xhr, status, error) {
                $("#contactForm").LoadingOverlay("hide", true);
                var errorAlert = '<div class="alert alert-danger alert-dismissible fade show" role="alert">' +
                '<span><i class="bi bi-x-octagon-fill me-2"></i>Something went wrong!</span>' +
                '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
                $("#alertBox").html(errorAlert);
            }
        });
    });
});