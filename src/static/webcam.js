//
window.addEventListener('DOMContentLoaded', () => {
	Webcam.set({
		width: 320,
		height: 240,
		image_format: 'jpeg',
		jpeg_quality: 90,
	});

	Webcam.attach('#webcam-container-1');
	document.querySelector('#webcam-container-1').style = 'width: 100%;';
	document.querySelector('#webcam-container-1>video').style = 'width:100%;';
	document.querySelector('#create-profile-form').addEventListener('submit', submitProfile);
	document.querySelector('#recognize-profile-form').addEventListener('submit', recognizeProfile);
	document.querySelector('#test-model').addEventListener('click', () => slideView(3));
});

const submitProfile = async (event) => {
	event.preventDefault();

	const captures = [];

	for (let i = 0; i < 2; i++) {
		capture = captureImage();
		captures.push(capture);
		await sleep(500);
	}

	slideView(1);
	await transferImages(captures);
	Webcam.attach('#webcam-container-2');
	document.querySelector('#webcam-container-2').style = 'width: 100%;';
	document.querySelector('#webcam-container-2>video').style = 'width:100%;';
	slideView(2);
};

const recognizeProfile = async (event) => {
	event.preventDefault();
	capture = captureImage();
	slideView(4);

	const image = getBase64(capture);
	const body = { name, image };

	const response = await post('/recognize-profile', body);
	console.log('response', response);

	slideView(5);
};

//
const captureImage = () => {
	let capture;
	Webcam.snap((data_uri) => (capture = data_uri));
	return capture;
};

//
const transferImages = async (captures) => {
	const images = captures.map((capture) => getBase64(capture));
	const name = document.querySelector('#name-input');
	const body = { name, images };

	const response = await post('/create-profile', body);
	console.log('response', response);
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
