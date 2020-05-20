PYTHON_INTERPRETER = python

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

atari-generate:
	$(PYTHON_INTERPRETER) src/atari/generator.py

atari-train:
	$(PYTHON_INTERPRETER) src/atari/offline.py

atari-autoencoder:
	$(PYTHON_INTERPRETER) src/atari/autoencoder.py
	
lunar-generate:
	$(PYTHON_INTERPRETER) src/lunar/generator.py

lunar-train:
	$(PYTHON_INTERPRETER) src/lunar/offline.py