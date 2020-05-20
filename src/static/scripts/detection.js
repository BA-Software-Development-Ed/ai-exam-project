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

const detectImage = async (event) => {
	event.preventDefault();
	const image = event.target['image-input'];
	const reader = new FileReader();

	reader.onloadend = async () => {
		const base64_source = reader.result.replace(/^data:.+;base64,/, '');

		console.log('loading...');

		const response = await postFetch('/detection', { image: base64_source });

		const image = document.querySelector('#image-detections');

		const image_source = `data:image/jpeg;base64,${response.image}`;

		image.src = image_source;
	};

	await reader.readAsDataURL(image.files[0]);
};

window.addEventListener('DOMContentLoaded', () => {
	document.querySelector('#detect-form').addEventListener('submit', detectImage);
});
