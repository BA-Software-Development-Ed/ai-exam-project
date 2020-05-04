const SERVER_URL = 'http://localhost:5000/';

//
const captureImages = async () => {
	const captures = [];

	for (let i = 0; i < 2; i++) {
		capture = captureImage();
		captures.push(capture);
		await sleep(500);
	}

	slideView(5);
	await transferImages(captures);

	slideView(5); // displays new slide after model trained
};

//
const transferImages = async (captures) => {
	const images = captures.map((capture) => getBase64(capture));
	const name = document.querySelector('#name-input');
	const body = { name, images };

	const response = await postFetch('/create-profile', body);
	console.log('response', response);
};

//
const setEventListeners = () => {
	const captureButton = document.querySelector('#capture-images-button');
	captureButton.onclick = captureImages;

	const nameButton = document.querySelector('#submit-name-button');
	nameButton.onclick = submitName;
};

//
const submitName = () => {
	const nameInput = document.querySelector('#name-input');

	if (nameInput.value) slideView(5);
	else alert('Name is required');
};

//
window.addEventListener('DOMContentLoaded', () => {
	setEventListeners();
});
