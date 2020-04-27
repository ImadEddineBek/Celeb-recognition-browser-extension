// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

'use strict'

// Selects saveImagesOption checkbox element
let saveImagesOption = document.getElementById('save_images');
// Selects thumbnailOption checkbox element
let thumbnailOption = document.getElementById('thumbnails');
let type_index = 'celebrities';

function setCheckbox(data, checkbox) {
    checkbox.checked = data;
};
chrome.storage.local.get('type', function (data) {
    var bkg = chrome.extension.getBackgroundPage();
    // bkg.console.log(data);
    type_index = data.type

});
// Gets thumbnails and saveImages value from storage
chrome.storage.local.get(['saveImages', 'thumbnails'], function (data) {
    setCheckbox(data.saveImages, saveImagesOption);
    saveImagesOption.checked = data.saveImages === true;
    setCheckbox(data.thumbnails, thumbnailOption);
});

// Saves users prefrences
function storeOption(optionName, optionValue) {
    let data = {};
    data[optionName] = optionValue;
    chrome.storage.local.set(data);
};

saveImagesOption.onchange = function () {
    storeOption('saveImages', saveImagesOption.checked);
};

thumbnailOption.onchange = function () {
    storeOption('thumbnails', thumbnailOption.checked);
};

let savedImages = document.getElementById('savedImages');

let deleteButton = document.getElementById('delete_button');

deleteButton.onclick = function () {
    let blankArray = [];
    chrome.storage.local.set({'savedImages': blankArray});
    location.reload();
};
var i = 1;
storeOption('i', 1);
// Gets saved downloaded images from storage
chrome.storage.local.get('savedImages', function (element) {
    let pageImages = element.savedImages;
    pageImages.forEach(function (image) {
        let hvdiv = document.createElement('div');
        hvdiv.className = "hvrbox";

        let newImage = document.createElement('img');
        newImage.className = 'hvrbox-layer_bottom';
        newImage.src = image;
        newImage.style.width = '100%';

        const Http = new XMLHttpRequest();

        const url = 'http://0.0.0.0:5000/who?type='.concat(type_index).concat('&src=').concat(encodeURIComponent(image));
        var bkg = chrome.extension.getBackgroundPage();
        // bkg.console.log(url);
        Http.open("GET", url);
        Http.send();

        let hvboxdiv = document.createElement('div');
        hvboxdiv.className = 'hvrbox-layer_top';
        let hrtext = document.createElement('div');
        hrtext.className = 'hvrbox-text';
        hrtext.innerText = 'unknown yet';

        Http.onreadystatechange = (e) => {
            var bkg = chrome.extension.getBackgroundPage();
            // bkg.console.log(Http);
            if (Http.responseText.length < 2) {
                hrtext.innerText = "Server Side issues, It's OK bro"
            } else {
                hrtext.innerText = Http.responseText
            }
        };

        hvboxdiv.appendChild(hrtext);

        hvdiv.appendChild(newImage);
        hvdiv.appendChild(hvboxdiv);
        newImage.style.width = '100%';
        hvdiv.style.width = '100%';
        hvdiv.addEventListener('click', function () {
            var bkg = chrome.extension.getBackgroundPage();
            // bkg.console.log(image);
            var x = document.getElementById("myGrid");
            if (x.className === "w3-row") {
                x.className = "w3-row-padding";
            } else {
                x.className = x.className.replace("w3-row-padding", "w3-row");
            }
        });
        let grid = document.getElementById('grid-' + i);
        grid.appendChild(hvdiv);

        if (i === 3) {
            i = 1;
        } else {
            i = i + 1
        }
    });
});
