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

// Number of classes to classify
const NUM_CLASSES = 4;

// Icons array for dynamically generated buttons
const iconsArr = [
    "/img/icon_play.svg",
    "/img/icon_pause.svg",
    "/img/icon_snd_on.svg",
    "/img/icon_snd_off.svg"
];
const musicTitlesArray = [
    "Track 1 Title: ",
    "Track 2 Title: ",
    "Track 3 Title: ",
];

// Webcam image size. Must be 227. 
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

// Add media file
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
        this.video = document.getElementById("input_video");
        this.video.setAttribute('autoplay', '');
        this.video.setAttribute('playsinline', '');
        document.getElementById('output-selectors').querySelector('span').innerText = musicTitlesArray[0];

        // Add video element to DOM
        document.body.appendChild(this.video);

        // Create training buttons and labels   
        for (let i = 0; i < NUM_CLASSES; i++) {
            const li = document.createElement('li');
            document.getElementById("trainingBtns").appendChild(li);
            li.style.padding = '10px';
            li.id = "elem" + i;

            const button = document.createElement('button');
            const img = document.createElement('img');
            img.src = iconsArr[i];
            li.appendChild(button);
            button.appendChild(img);

            // Listen for button click (i.e. user is training actions 0 - 3) 
            button.addEventListener('mousedown', () => this.training = i);
            button.addEventListener('mouseup', () => this.training = -1);

            // Setup media thumbnail buttons
            const vid1 = "/media/tester.mp4";
            const vid2 = "/media/flowers.mp4";
            const vid3 = "/media/tester.mp4";

            const container = document.getElementById('output_video');
            const source = document.getElementById('mp4video');
            const title = document.getElementById('music-credit');

            // User clicked first thumbnail
            out1.addEventListener('click', function(event) {
                source.setAttribute('src', vid1)
                container.load(source);
                title.innerText = musicTitlesArray[0];
            });
            // User clicked second thumbnail
            out2.addEventListener("click", function(event) {
                source.setAttribute("src", vid2);
                container.load(source);
                title.innerText = musicTitlesArray[1];
            });
            // User clicked third thumnail
            out3.addEventListener("click", function(event) {
                source.setAttribute("src", vid3);
                container.load(source);
                title.innerText = musicTitlesArray[2];
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

                    // Update info text
                    if (exampleCount[i] > 0) {
                        this.infoTexts[i].innerText = ` ${exampleCount[i]} examples`;
                        document.getElementById('meter-text' + i).style.width = ` ${res.confidences[i] * 100}%`;
                    }
                    // If Confidence score > 50% 'light up' relevant btn and display meter value
                    // Play/Pause and UnMute/Mute function as toggles - so we need to set color of 
                    // opposite toggle back to (default) blue
                    if (res.classIndex == 0 && res.confidences[i] >= .8) {
                        document.getElementById("output_video").play();
                        document.getElementById("elem0").querySelector('button').style.backgroundColor = "orange";
                        document.getElementById("elem1").querySelector('button').style.backgroundColor = "#35D9FE";
                    } else if (res.classIndex == 1 && res.confidences[i] >= .8) {
                        document.getElementById("output_video").pause();
                        document.getElementById("elem1").querySelector('button').style.backgroundColor = "orange";
                        document.getElementById("elem0").querySelector('button').style.backgroundColor = "#35D9FE";
                    } else if (res.classIndex == 2 && res.confidences[i] >= .8) {
                        document.getElementById("output_video").volume = 1;
                        document.getElementById("output_video").muted = false;
                        document.getElementById("elem2").querySelector('button').style.backgroundColor = "orange";
                        document.getElementById("elem3").querySelector('button').style.backgroundColor = "#35D9FE";
                    } else if (res.classIndex == 3 && res.confidences[i] >= .8) {
                        document.getElementById("output_video").muted = true;
                        document.getElementById("elem3").querySelector('button').style.backgroundColor = "orange";
                        document.getElementById("elem2").querySelector('button').style.backgroundColor = "#35D9FE";
                    }
                }
            }

            // Dispose of images when done
            image.dispose();
            if (logits != null) {
                logits.dispose();
            }
        }
        this.timer = requestAnimationFrame(this.animate.bind(this));
    }
}

window.addEventListener('load', () => new Main());