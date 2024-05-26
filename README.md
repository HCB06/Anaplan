# PyerualJetwork 
--- UPDATE 1.2.6 FEATURES ---

GetWieghts(), GetPreds(), GetAcc() and GetDf() funcs added.

for ex.:

Train = plan.TrainPLAN(.......)

W = Train[plan.GetWeights()]



Test = plan.TestPLAN(.......)

TestPreds = Test[plan.GetPreds()]

TestAcc = Test[plan.GetAcc()]




LoadedModel = plan.LoadPLAN(.......)

df = LoadedModel[plan.GetDf()]


--- UPDATE 1.2.6 FEATURES ---

Pyerual Jetwork is a machine learning library written in Python for
professionals, incorporating advanced, unique, new, and modern
techniques. Its most important component is the PLAN (Pruning Learning
Artificial Neural Network).

Both the PLAN algorithm and the Pyerual Jetwork library were created
by Hasan Can Beydili, and all rights are reserved by Hasan Can Beydili.

As of 05/24/2024, the library includes only the PLAN module, but other
machine learning modules are expected to be added in the future.

The PLAN algorithm will not be explained in this document. This
document focuses on how professionals can integrate and use Pyerual
Jetwork in their systems. However, briefly, the PLAN algorithm can be
described as a classification algorithm. For more detailed information, you
can check out my Medium article: https://medium.com/@tchasancan/new-
artificial-neural-network-architecture-c11edacb06fa

The functions of the Pyerual Jetwork modules, except for specific
names like 'PLAN', use PascalCase naming convention.
The PLAN module consists of 16 functions. Of these, 7 are main
functions (which the user will interact with frequently), and the
remaining 9 are auxiliary functions.

HOW DO I IMPORT IT TO MY PROJECT?

Anaconda users can access the 'Anaconda Prompt' terminal from the Start
menu and add the necessary library modules to the Python module search
queue by typing "pip install pyerualjetwork" and pressing enter. If you are
not using Anaconda, you can simply open the 'cmd' Windows command
terminal from the Start menu and type "pip install pyerualjetwork". After
installation, it's important to periodically open the terminal of the
environment you are using and stay up to date by using the command "pip
install pyerualjetwork --upgrade". As of 05/24/2024, the most current
version is "1.2.0". This version is the successor to version "1.0" of the
library. Previous versions were for testing purposes and are non-functional.
After installing the modules using "pip" you can now call the library
modules in your project environment. To do this, simply write "import
plan" in the project where you're writing your code. Alternatively, you can
use "import plan as pl". Now, you can call the necessary functions from the
plan module.



MAIN FUNCTIONS:
1. TrainPLAN(9Args)
2. TestPLAN(8Args)
3. SavePLAN(14Args)
4. LoadPLAN(3Args)
5. PredictFromDiscPLAN(4Args)
6. PredictFromRamPLAN(7Args)
7. AutoBalancer(3Args)

   
1.) TrainPLAN(TrainInputs, TrainLabels, ClassCount, Layers,
Neurons, ThresholdSigns, ThresholdValues, Normalizations,
Activations)

The purpose of this function, as the name suggests, is to train the
model.

a. TrainInputs: A list consisting of input vectors or
matrices representing training examples.
b. TrainLabels: One-hot encoded list of educational labels.
c. ClassCount: Total class count.
d. Layers: A list consisting of layer types 'fex' (Feature
Extraction Layer) or 'cat' (Catalyser Layer) also
represents the number of layers in the model.
e. Neurons: A numerical list indicating the number of
neurons in each layer (starting with the first layer at the
beginning and ending with the last layer at the end).
f. ThresholdSigns: A PLAN algorithm-specific
thresholding metric. ‘==’ , ’!=’, ‘<’, ‘>’ ,‘none’ (gradient
descent and epoch approach are not available in PLAN
models. Therefore, instead of epoch, error function and
optimization metrics, it is one of the new metrics specific
to PLAN. It specifies the sign of the threshold to be used
in the layers sequentially in a list.)(Hyperparameter)
g. ThresholdValues: A numerically ordered list of values
that will be placed next to the threshold sign.
(Hyperparameter)
h. Normalizations: A sequentially ordered list containing
'y' or 'n' values indicating whether data normalization
will be applied to each layer. (Normalization divides
each element in the data vector by the element with the
largest absolute value to produce an output consisting of
1s and 0s.)
i. Activations: A sequentially ordered list that can
currently take a total of 4 values indicating which
activation function will be used in each layer. 'sigmoid',
'relu', 'softmax' 'none'.

The outputs of this function are, in order: a list of trained weight matrices,
a list of class equivalents of training predictions, and a training accuracy
rate.

2.) TestPLAN(TestInputs, TestLabels, Layers, ThresholdSigns,
ThresholdValues, Normalizations, Activations,W)

This function calculates the test accuracy of the model using the inputs and
labels set aside for testing, along with the weight matrices and other model
parameters obtained as output from the training function.
a. W: A list of numpy weight matrices, where each element is a
numpy weight matrix. (Descriptions of other parameters are
the same as Function 1)
The outputs of this function are, in order: a list of class equivalents of test
predictions, and a test accuracy rate.

3.) SavePLAN(ModelName, ModelTyepe, ClassCount,
ThresholdSigns, ThresholdValues, Normalizations,
Activations,TestAcc,LogType,WeightsType,WeightsFormat,Save
Path,W)

This function creates log files in the form of a pandas DataFrame
containing all the parameters and information of the trained and tested
model, and saves them to the specified location along with the weight
matrices.

a. ModelName: Specify the name of the model to be saved, e.g.:
'Model1'.
b. ModelType: Specify the type of the model to be saved, e.g.:
'PLAN'.
c. TestAcc: Test accuracy rate.
d. LogType: Specify the file extension of the log file to be saved,
e.g.: 'txt' or 'csv'.
e. WeightsType: Specify the file extension of the weight matrices
to be saved, e.g.: 'txt' or 'mat'.
f. WeightsFormat: Specify the numeric data type of the weight
matrices to be saved, e.g.: 'd' = integer, 'f' = floating-point, 'raw'
= saves as is.
g. SavePath: Specify the file path where the model will be saved.
(Note: Use '/' when specifying the location and use '/' again at
the end.) (W and other parameters have been mentioned
above.)

This function returns messages such as 'saved' or 'could not be saved' as
output.

4.) LoadPLAN(ModelName, LoadPath,LogType)

This function retrieves everything about the model into the Python
environment from the saved log file and the model name.

a. LoadPath: Specify the file path where the model is saved.
(Note: Use '/' when specifying the location and use '/' again
at the end.) (Other parameters have been mentioned.)
This function returns the following outputs in order: W, Layers,
ThresholdSigns, ThresholdValues, Normalizations, Activations, and the
data frame of the loaded model as the final output."

5.) PredictFromDiscPLAN(Input, ModelName, ModelPath,
LogType)

This function loads the model directly from its saved location, predicts a
requested input, and returns the output. (It can be integrated into
application systems and the output can be converted to .json format and
used in web applications.)

a. Input: The real-world example to be predicted.
b. ModelPath: The file path where the model is located (Note:
Use '/' when specifying the location and use '/' again at the
end.)( (Other parameters are information about the model
and are defined as described and listed above.)

This function returns the last output layer of the model as the output of the
given input.

6.) PredictFromRamPLAN(Input, Layers, ThresholdSigns,
ThresholdValues, Normalizations, Activations,W)

This function predicts and returns the output for a requested input using a
model that has already been loaded into the program (located in the
computer's RAM). (It can be integrated into application systems and the
output can be converted to .json format and used in web applications.)
(Other parameters are information about the model and are defined as
described and listed above.)

This function returns the last output layer of the model as the output of the
given input.

7.) AutoBalancer(TrainInputs, TrainLabels, ClassCount)

This function aims to balance all training data according to class
distribution before training the model. All data is reduced to the number of
data points of the class with the least number of examples. (Sometimes it
improves performance in PLAN models, and sometimes it decreases it.)
(Parameter descriptions are given in the first function.)
This function returns the following outputs in order: a list containing the
balanced training data and a list containing the balanced training labels.
As additional information, you can use the train_split methods in the scikit-
learn library before training. Especially, selecting your training data from
higher quality data that can generalize better using the 'random_state'
parameter is a factor that directly affects model performance. In addition,
functions that will perform such training modification operations will also
be included in the PLAN module in the next update.

ERROR MESSAGES:

100 Type Errors: Value Errors.
200 Type Errors: Variable type errors.
300 Type Errors: List length errors.

LAST PART:

Despite being in its early stages of development, Pyerual Jetwork has
already demonstrated its potential to deliver valuable services and
solutions in the field of machine learning. Notably, it stands as the first
library dedicated to Plan (Pruning Learning Artificial Neural
Network), embracing innovation and welcoming new ideas from its
users with open arms. Recognizing the value of diverse perspectives
and fresh ideas, I, Hasan Can Beydili, the creator of Pyerual Jetwork,
am committed to fostering an open and collaborative environment
where users can freely share their thoughts and suggestions. The most
promising contributions will be carefully considered and potentially
integrated into the Pyerual Jetwork library. For your suggestions, lists
and feedback, my e-mail address is: tchasancan@gmail.com
And finally, trust the PLAN…
