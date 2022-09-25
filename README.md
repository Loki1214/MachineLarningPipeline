# MachineLarningPipeline

This machine larning pipe line consistes of the following services.

 - webapp
	- Web application (runs at http://localhost:5050) which classifies the digit written in the uploaded image.

	- Accepts the feedback about the correct label from the client.

	- If the feedback is given, the application registers it to the *database* and stores the uploaded image to the *storage*.

 - training
	- Trains the neural network which the *webapp* uses to classify digit images.

	- It periodically checks how many unused images (i.e., images that have not been used for training yet) are accumulated in the database.
	If the number of the new images excess the threshold (it is set to be 10 by default), it starts training the nerural network.
	(cf. [MachineLarning/train_NeuralNetwork.sh](MachineLarning/train_NeuralNetwork.sh))

	- Each time the training finishes, it sends a request for the *appbuilder* to build a new docker image from which the *webapp* container will be created.

	- Definition of the neural network is given in the class *Net* in [MachineLarning/work/model_definition.py](MachineLarning/work/model_definition.py).

	- The neural network should accept an PIL image and returns the classification result as an integer.

 - appbuilder
	- Builds a docker image for the *webapp*.

	- It receives three files 'model_definition.py', 'model_weights.pth', and 'imageClassifier.py' via http, and then builds a docker image from them.
	For example, when the *training* container completes the training, it sends a request for building a new image which uses the newly trained neural network via the following command:
	````` sh
		# runs at the 'training' container
		curl -X POST
			-F  imageClassifier.py=@trainedDNN/imageClassifier.py  \
			-F model_definition.py=@trainedDNN/model_definition.py \
			-F   model_weights.pth=@trainedDNN/model_weights.pth   \
			http://appbuilder:5000 2>/dev/null
	`````
	- 'imageClassifier.py' provides the *ImageClassifier* class with the *predict* method, which accepts a PIL image or a torch tensor and returns the classification result as an integer.

	- 'model_definition.py' and 'model_weights.py' are refered by 'imageClassifier.py'.


 - storage
	- The data storage which stores the digit images

	- Information needed to access the *storage* should be given in [common-variables.env](common-variables.env) as environment variables.

	- These environment variables are then refered by the python scripts that need to connect the *storage*.
	(cf. [MachineLarning/work/storage/settings.py](MachineLarning/work/database/settings.py),
	[WebAppBuilder/storage/settings.py](MachineLarning/work/database/settings.py))

	- In the current verson, we adopt 'minio'.

	- The storage console can be accessed from [http://localhost:9001](http://localhost:9001)


 - database
	- The database which stores the information (such as pass to the image file and the correct label) about the images to be used to train the nerural network.

	- Information needed to access the database should be given in [common-variables.env](common-variables.env) as environment variables.

	- These environment variables are then refered by the python scripts that need to connect the *database*.
	(cf. [MachineLarning/work/database/settings.py](MachineLarning/work/database/settings.py),
	[WebAppBuilder/database/settings.py](MachineLarning/work/database/settings.py))

	- In the current verson, we adopt 'mysql'.

	- The database console can be accessed from [http://localhost:8080](http://localhost:8080)

## Running the pipeline

1. If you are using a local registry, set the address of your registry to the variable 'LOCAL_REGISTRY' in the [.env](.env) file.
If you want to use the dafault registry, leave it blank.
	- You can use [Registry/docker-compose.yml](Registry/docker-compose.yml) to launch a local docker registry which runs at 'localhost:5050'.

2. If the image 'mywebapp' for the 'webapp' container does not exist at all in your registry, build the first version of it using the pretrained neural network in [WebAppBuilder/pretrainedDNN](WebAppBuilder/pretrainedDNN) with the following command;
````` sh
	cd WebAppBuilder
	./build_new_image.sh pretrainedDNN
	cd -
`````

3. Finally, Run
````` sh
	# run at the project root directory
	docker compose up
`````
- This command creates docker volumes named 'mlpl-database-volume' and 'mlpl-storage-volume'.

- If the MNIST dataset is not registered to the database and the storage with these docker volumes, the *training* container automatically downloads and registers MNIST to the database and the storage.


---
## Proxy settings
- If you are running this machine larning pipeline behind a proxy, you need to set enviromnent variables ('HTTP_PROXY', 'HTTPS_PROXY', and 'NO_PROXY') in your shell or in [.env](.env) file.

- If these environment variables are set in your shell, [docker-compose.yml](docker-compose.yml) automatically passes it to the build context of *training* and *appbuilder*, so you need not to set them in [.env](.env) file.