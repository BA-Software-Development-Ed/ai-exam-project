//
const post = async (uri, body) => {
	const options = {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	};

	const response = await fetch(uri, options);
	const json = await response.json();
	return json;
};

const submit = async (event) => {
	event.preventDefault();
	const image = event.target['image-input'];

	if (!image.value) {
		// set submit button as error
		const button = document.querySelector('.submit-button');
		button.innerText = 'image required!';
		button.style.color = '#fff';
		button.style.backgroundColor = '#FF3062';
		return;
	}

	slideView(1);

	const form = document.querySelector('form');
	const uri = form.dataset.uri;
	const reader = new FileReader();

	reader.onloadend = async () => {
		const base64_source = reader.result.replace(/^data:.+;base64,/, '');
		const response = await post(uri, { image: base64_source });
		const image = document.querySelector('#image-detections');
		const image_source = `data:image/jpeg;base64,${response.image}`;
		image.src = image_source;

		slideView(2);
	};

	await reader.readAsDataURL(image.files[0]);
};

const read_file = (input) => {
	if (input.files && input.files[0]) {
		const reader = new FileReader();

		reader.onload = function (event) {
			const image = document.querySelector('#image-preview');
			image.src = event.target.result;
		};

		reader.readAsDataURL(input.files[0]);

		// resets submit button
		const button = document.querySelector('.submit-button');
		button.innerText = 'submit image';
		button.style.color = '#fff';
		button.style.backgroundColor = 'dodgerblue';
	}
};

window.addEventListener('DOMContentLoaded', () => {
	document.querySelector('#detect-form').addEventListener('submit', submit);
});

// utility
const slideView = (slide) => {
	const main = document.querySelector('main');
	const unloaded = document.querySelectorAll('.unloaded');
	const offset = 0 - 100 * slide;

	main.style.left = `${offset}vw`;
	unloaded[0].className = '';
};
