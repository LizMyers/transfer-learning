// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

const $ = document.querySelector.bind(document);
const $$ = document.querySelectorAll.bind(document);

/*------------
// Change the background color of a class.
$('.class').style.background = "#BADA55";

// Change the inner HTML of an ID.
$('#id').innerHTML = "<span>Cool beans!</span>";

// Select all images on the webpage.
$$('img')

// Print the image addresses for all the images on a webpage.
$$('img').forEach(img => console.log(img.src))
-------------*/

// Number of classes to classify
const NUM_CLASSES = 4;
const labelsArr = ["Play", "Pause", "Vol 100%", "Mute"];
const iconsArr = [
    "/img/icon_play.svg",
    "/img/icon_pause.svg",
    "/img/icon_snd_on.svg",
    "/img/icon_snd_off.svg"
];
// Webcam Image size. Must be 227. 
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

// Add Media File
// const audio = new Audio('media/doorbell_mono.wav');
const output_video = document.getElementById("output_video");


class Main {
    constructor() {
        // Initiate variables
        this.infoTexts = [];
        this.confScore = 0;
        this.training = -1; // -1 when no class is being trained
        this.videoPlaying = false;
        this.output_videoPlaying = false;

        // Initiate deeplearn.js math and knn classifier objects
        this.bindPage();

        // Create video element that will contain the webcam image
        //this.video = document.createElement('video');
        this.video = document.getElementById("input_video");
        this.video.setAttribute('autoplay', '');
        this.video.setAttribute('playsinline', '');

        // Add video element to DOM
        document.body.appendChild(this.video);


        // Create training buttons and info texts    
        for (let i = 0; i < NUM_CLASSES; i++) {
            const li = document.createElement('li');
            document.getElementById("trainingBtns").appendChild(li);
            li.style.padding = '10px';
            li.id = "elem" + i;

            // Create training button
            const button = document.createElement('button');
            const img = document.createElement('img');
            img.src = iconsArr[i];
            li.appendChild(button);
            button.appendChild(img);

            // Listen for mouse events when clicking the button
            button.addEventListener('mousedown', () => this.training = i);
            button.addEventListener('mouseup', () => this.training = -1);

            //setup thumbnails
            const vid1 = "/resources/mp4/tester.mp4";
            const vid2 = "/resources/mp4/flowers.mp4";
            const vid3 = "/resources/mp4/tester.mp4";
            const container = document.getElementById('output_video');
            const source = document.getElementById('mp4video');

            out1.addEventListener('click', function(event) {
                source.setAttribute('src', vid1)
                container.load(source);
            });


            out2.addEventListener("click", function(event) {
                source.setAttribute("src", vid2);
                container.load(source);
            });
            out3.addEventListener("click", function(event) {
                source.setAttribute("src", vid3);
                container.load(source);
            });


            // Create info text
            const infoText = document.createElement('span');
            infoText.innerText = " No examples added";
            li.appendChild(infoText);
            this.infoTexts.push(infoText);

            // Create confidence meter
            const meter = document.createElement('div');
            const meterText = document.createElement('span');

            li.appendChild(meter);
            meter.id = "meter" + i;
            meter.className = "meter";
            meter.appendChild(meterText);
            meterText.id = "meter-text" + i;
            meterText.className = "meterText";
        }

        // Setup webcam
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then((stream) => {
                this.video.srcObject = stream;
                this.video.width = IMAGE_SIZE;
                this.video.height = IMAGE_SIZE;

                this.video.addEventListener('playing', () => this.videoPlaying = true);
                this.video.addEventListener('paused', () => this.videoPlaying = false);

                //flip the input video
                this.video.style.transform = 'scale(-1, 1)';
            })
    }

    async bindPage() {
        this.knn = knnClassifier.create();
        this.mobilenet = await mobilenetModule.load();

        this.start();
    }

    start() {
        if (this.timer) {
            this.stop();
        }
        this.video.play();
        this.timer = requestAnimationFrame(this.animate.bind(this));
    }

    stop() {
        this.video.pause();
        cancelAnimationFrame(this.timer);
    }

    async animate() {
        if (this.videoPlaying) {
            // Get image data from video element
            const image = tf.fromPixels(this.video);

            let logits;
            // 'conv_preds' is the logits activation of MobileNet.
            const infer = () => this.mobilenet.infer(image, 'conv_preds');

            // Train class if one of the buttons is held down
            if (this.training != -1) {
                logits = infer();

                // Add current image to classifier
                this.knn.addExample(logits, this.training)
            }

            const numClasses = this.knn.getNumClasses();
            if (numClasses > 0) {

                // If classes have been added run predict
                logits = infer();
                const res = await this.knn.predictClass(logits, TOPK);

                for (let i = 0; i < NUM_CLASSES; i++) {

                    // The number of examples for each class
                    const exampleCount = this.knn.getClassExampleCount();

                    // Make the predicted class bold
                    if (res.classIndex == i) {
                        this.infoTexts[i].style.fontWeight = 'bold';
                    } else {
                        this.infoTexts[i].style.fontWeight = 'normal';
                    }

                    // Update info text
                    if (exampleCount[i] > 0) {
                        this.infoTexts[i].innerText = ` ${exampleCount[i]} examples`;
                        document.getElementById('meter-text' + i).style.width = ` ${res.confidences[i] * 100}%`;
                    }
                    //Do Something Based On Prediction
                    if (res.classIndex == 0 && res.confidences[i] >= .5) {
                        console.log("Play the Video");
                        document.getElementById("output_video").play();
                        document.getElementById("elem0").querySelector('button').style.backgroundColor = "orange";
                        document.getElementById("elem1").querySelector('button').style.backgroundColor = "#35D9FE";
                    } else if (res.classIndex == 1 && res.confidences[i] >= .5) {
                        console.log("Pause the Video");
                        document.getElementById("output_video").pause();
                        document.getElementById("elem1").querySelector('button').style.backgroundColor = "orange";
                        document.getElementById("elem0").querySelector('button').style.backgroundColor = "#35D9FE";
                    } else if (res.classIndex == 2 && res.confidences[i] >= .5) {
                        console.log("Volume = 100%");
                        document.getElementById("output_video").volume = 1;
                        document.getElementById("output_video").muted = false;
                        document.getElementById("elem2").querySelector('button').style.backgroundColor = "orange";
                        document.getElementById("elem3").querySelector('button').style.backgroundColor = "#35D9FE";
                    } else if (res.classIndex == 3 && res.confidences[i] >= .5) {
                        console.log("Mute the Audio")
                        document.getElementById("output_video").muted = true;
                        document.getElementById("elem3").querySelector('button').style.backgroundColor = "orange";
                        document.getElementById("elem2").querySelector('button').style.backgroundColor = "#35D9FE";
                    }
                }
            }

            // Dispose image when done
            image.dispose();
            if (logits != null) {
                logits.dispose();
            }
        }
        this.timer = requestAnimationFrame(this.animate.bind(this));
    }
}

window.addEventListener('load', () => new Main());