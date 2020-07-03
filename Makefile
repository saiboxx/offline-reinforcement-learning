PYTHON_INTERPRETER = python

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

lunar-generate:
	$(PYTHON_INTERPRETER) src/lunar/generator.py

lunar-train:
	$(PYTHON_INTERPRETER) src/lunar/offline.py

lunar-inference:
	$(PYTHON_INTERPRETER) src/lunar/inference.py

atari-generate:
    echo "WARNING: The Atari branch hasn't been maintained. Functions can be deprecated."
	$(PYTHON_INTERPRETER) src/atari_archive/generator.py

atari-train:
    echo "WARNING: The Atari branch hasn't been maintained. Functions can be deprecated."
	$(PYTHON_INTERPRETER) src/atari_archive/offline.py

atari-autoencoder:
    echo "WARNING: The Atari branch hasn't been maintained. Functions can be deprecated."
	$(PYTHON_INTERPRETER) src/atari_archive/autoencoder.py