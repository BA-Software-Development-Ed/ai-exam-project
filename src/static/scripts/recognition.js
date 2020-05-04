//
const recognizeImage = async () => {
	slideView(3);
	const capture = await captureImage();
	//const response = await postFetch('', { capture });
	await sleep(3000); // simulates loading
	slideView(3);
};

//
const setEventListeners = () => {
	const captureButton = document.querySelector('#capture-image-button');
	captureButton.onclick = recognizeImage;
};

//
window.addEventListener('DOMContentLoaded', () => {
	setEventListeners();
});
