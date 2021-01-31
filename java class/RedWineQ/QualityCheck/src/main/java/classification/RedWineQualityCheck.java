package classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RedWineQualityCheck {
    private static final int SEED = 567;
    private static final int INPUT = 11;
    private static final int OUTPUT = 11;
    private static final int EPOCH = 500;
    private static final double LR = 1e-4;
    private static final double SR = 0.8;

    public static void main(String[] args) throws Exception{

        File file = new ClassPathResource("winequality/NewWineSMOTE.csv").getFile();
        FileSplit fileSplit = new FileSplit(file);

        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(fileSplit);
//=========================================================================
        //  Step 1 : Build Schema
//=========================================================================
        Schema ss = new Schema.Builder()
                .addColumnsDouble("fixed acidity", "volatile acidity", "citric acid", "residual sugar"
                        , "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH"
                        , "sulphates", "alcohol")
                .addColumnCategorical("quality", Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"))
                .build();
//=========================================================================
        //  Step 2 : Build TransformProcess to transform the data
//=========================================================================
        TransformProcess tp = new TransformProcess.Builder(ss)
//                .convertToInteger("quality")
                .build();

        List<List<Writable>> rawList = new ArrayList<>();

        while (rr.hasNext()){
            rawList.add(rr.next());
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(rawList, tp);
//========================================================================
        //  Step 3 : Create Iterator ,splitting trainData and testData
//========================================================================
        RecordReader cc = new CollectionRecordReader(processedData);
        DataSetIterator dataIterator = new RecordReaderDataSetIterator(cc, rawList.size(), -1, 11);

        DataSet allData = dataIterator.next();
        allData.shuffle(SEED);

        SplitTestAndTrain splitData = allData.splitTestAndTrain(SR);

        DataSet trainData = splitData.getTrain();
        DataSet testData = splitData.getTest();

        System.out.println("Training vector : ");
        System.out.println(Arrays.toString(trainData.getFeatures().shape()));
        System.out.println("Test vector : ");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));
//========================================================================
        //  Step 4 : DataNormalization
//========================================================================
        DataNormalization scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainData);
        scaler.transform(trainData);
        scaler.transform(testData);
//========================================================================
        //  Step 5 : MultiNetwork Configuration
//========================================================================
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(LR))
                .activation(Activation.LEAKYRELU)
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(INPUT)
                        .nOut(1024)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nOut(1024)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nOut(1024)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nOut(1024)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nOut(200)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .nOut(OUTPUT)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
//========================================================================
        //  Step 6 : Setup UI , listeners
//========================================================================
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        model.setListeners(new StatsListener(storage, 10), new ScoreIterationListener(10));
//========================================================================
        //  Step 7 : Training and Evaluation
//========================================================================
        for (int i = 0; i < EPOCH; i++){
            model.fit(trainData);
        }

        Evaluation evalTrain = model.evaluate(new ViewIterator(trainData, rawList.size()));
        Evaluation evalTest = model.evaluate(new ViewIterator(testData, rawList.size()));
        System.out.print("Train Data");
        System.out.println(evalTrain.stats());

        System.out.print("Test Data");
        System.out.print(evalTest.stats());
    }
}
