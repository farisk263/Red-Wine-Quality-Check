package classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.datavec.api.transform.schema.Schema;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;


import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static classification.PlotUtil.plotLossGraph;

public class RedWineKFold {

    private static final int SEED = 567;
    private static final int INPUT = 11;
    private static final int OUTPUT = 6;
    private static final int EPOCH = 500;
    private static final double LR = 0.001;

    static ArrayList<Double> trainingLoss = new ArrayList<>();
    static ArrayList<Double> validationLoss = new ArrayList<>();

    public static void main(String[] args) throws Exception{

        File file = new ClassPathResource("winequality/NewWineSMOTE.csv").getFile();
        FileSplit fileSplit = new FileSplit(file);

        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(fileSplit);

//=========================================================================
        //  Step 1 : Build Schemas
//=========================================================================

        Schema ss = new Schema.Builder()
                .addColumnsDouble("fixed acidity", "volatile acidity", "citric acid", "residual sugar"
                , "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH"
                , "sulphates", "alcohol")
                .addColumnCategorical("quality", Arrays.asList("3", "4", "5", "6", "7", "8"))
                .build();

//=========================================================================
        //  Step 2 : Build TransformProcess to transform the data
//=========================================================================

        TransformProcess tp = new TransformProcess.Builder(ss)
//                .convertToInteger("quality")
                .categoricalToInteger("quality")
                .build();

        List<List<Writable>> rawList = new ArrayList<>();

        while (rr.hasNext()){
            rawList.add(rr.next());
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(rawList, tp);

//========================================================================
        //  Step 3 : Create Dataset and shuffle the data
//========================================================================

        RecordReader cc = new CollectionRecordReader(processedData);
        DataSetIterator dataIterator = new RecordReaderDataSetIterator(cc, rawList.size(), -1, OUTPUT);

        DataSet allData = dataIterator.next();
        allData.shuffle(SEED);

//========================================================================
        //  Step 4 : MultiNetwork Configuration
//========================================================================


        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(LR))
                .activation(Activation.RELU)
                .l2(0.001)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(INPUT)
                        .nOut(256)
                        .build())
                .layer(new DropoutLayer(0.2))
                .layer(new DenseLayer.Builder()
                        .nOut(256)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(32)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nOut(OUTPUT)
                        .lossFunction(new LossMCXENT())
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

//========================================================================
        //  Step 5 : Initialize for K Fold Cross Validation Train/Test
//========================================================================

        KFoldIterator kFoldIterator = new KFoldIterator(5, allData);

        int i = 1;
        System.out.println("-------------------------------------------------------------");

        //initialize an empty list to store the F1 score
        ArrayList<Double> f1List = new ArrayList<>();

        //for each fold
        while (kFoldIterator.hasNext()) {

            System.out.println("\n\n\nFOLD: " + i + "\n\n");

            //for each fold, get the features and labels from training set and test set
            DataSet currDataSet = kFoldIterator.next();
            INDArray trainFoldFeatures = currDataSet.getFeatures();
            INDArray trainFoldLabels = currDataSet.getLabels();
            INDArray testFoldFeatures = kFoldIterator.testFold().getFeatures();
            INDArray testFoldLabels = kFoldIterator.testFold().getLabels();
            DataSet trainDataSet = new DataSet(trainFoldFeatures, trainFoldLabels);
            DataSet testDataSet = new DataSet(testFoldFeatures, testFoldLabels);

//========================================================================
            //  Step 6 : Data Normalization and set up iterator
//========================================================================

            DataNormalization scaler = new NormalizerStandardize();
            scaler.fit(trainDataSet);
            scaler.transform(trainDataSet);
            scaler.transform(testDataSet);

            DataSetIterator trainIter = new ViewIterator(trainDataSet, trainDataSet.numExamples());
            DataSetIterator testIter = new ViewIterator(testDataSet, testDataSet.numExamples());

            DataSetLossCalculator trainLossCalculator = new DataSetLossCalculator(trainIter, true);
            DataSetLossCalculator validLossCalculator = new DataSetLossCalculator(testIter, true);


//========================================================================
            //  Step 7 : Network Initialization
//========================================================================

            MultiLayerNetwork model = new MultiLayerNetwork(config);
            model.init();

//========================================================================
            //  Step 8 : Stats GUI & Loss Score Listener
//========================================================================

            StatsStorage storage = new InMemoryStatsStorage();
            UIServer server = UIServer.getInstance();
            server.attach(storage);

            model.setListeners(new StatsListener(storage, 10), new ScoreIterationListener(10));

//========================================================================
            //  Step 9 : Train fold & add Loss Score into arrays
//========================================================================

            //train the data
            for (int j = 0; j < EPOCH; j++) {
                model.fit(trainDataSet);

                trainingLoss.add(trainLossCalculator.calculateScore(model)); // calculate training loss and add to trainingLoss ArrayList
                validationLoss.add(validLossCalculator.calculateScore(model)); // calculate validation loss and add to validationLoss ArrayList

            }
//========================================================================
            //  Step 10 : Plot Training/Validation Loss Graph & evaluate the train/test set from current fold
//========================================================================

            plotLossGraph("Training/Validation Loss: Fold " + i,"Number of Epochs", "Training/Validation Loss", trainingLoss, validationLoss, EPOCH);

            Evaluation evalTrain = model.evaluate(trainIter);
            Evaluation evalTest = model.evaluate(testIter);

//========================================================================
            //  Step 11 : Print out evaluation metrics & clear out array for the next fold
//========================================================================

            //print out the train results
            System.out.println("\n\nTrain Data:\n" + evalTrain.stats());

            //print out the test results
            System.out.println("\n\nTest Data:\n" + evalTest.stats());
            //save the eval results
            f1List.add(evalTest.f1());

            trainingLoss.clear();
            validationLoss.clear();
            trainIter.reset();
            testIter.reset();

            i++;
            System.out.println("-------------------------------------------------------------");
        }

//========================================================================
        //  Step 12: Print out mean score for the folds
//========================================================================

        INDArray f1scores = Nd4j.create(f1List);
        System.out.println("Average Test Data's F1 scores for all folds: " + f1scores.mean(0));



    }

}
