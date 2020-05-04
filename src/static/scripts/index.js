//
const setWebcamStyle = () => {
	const webcamContainer = document.querySelector('#webcam-container');
	const webcamVideo = document.querySelector('#webcam-container>video');

	webcamContainer.style = 'flex: 1;';
	webcamVideo.style = 'width:100%;';
};

// note: set when needed
const setWebcam = () => {
	Webcam.set({
		width: 320,
		height: 240,
		image_format: 'jpeg',
		jpeg_quality: 90,
	});

	Webcam.attach('#webcam-container');
};

//
window.addEventListener('DOMContentLoaded', () => {
	setWebcam();
	setWebcamStyle();
});

/* global functions */

//
const captureImage = () => {
	let capture;
	Webcam.snap((data_uri) => (capture = data_uri));
	return capture;
};

//
const slideView = (slideAmounts) => {
	const container = document.querySelector('main');
	const unloaded = document.querySelectorAll('.unloaded');

	const loadedWidth = 100 * (slideAmounts - unloaded.length);
	const centerOffset = parseInt(slideAmounts / 2) * 100;
	const offset = centerOffset - loadedWidth;

	container.style.left = `${offset}vw`;
	unloaded[0].className = '';
};

//
const postFetch = async (uri, body) => {
	const options = {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	};

	const response = await fetch(uri, options);
	const json = await response.json();
	return json;
};

//
const getBase64 = (data_uri) => {
	const regex = new RegExp('data:image/jpeg;base64,(?<base64>.*)');
	const matches = regex.exec(data_uri);
	return matches.groups.base64;
};

//
function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}
