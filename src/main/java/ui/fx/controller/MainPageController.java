package ui.fx.controller;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;

import alg.GeneticAlgorithm;
import javafx.application.Platform;
import javafx.beans.value.ObservableValue;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.layout.Border;
import javafx.scene.layout.BorderStroke;
import javafx.scene.layout.BorderWidths;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import ui.fx.GeneticAlgorithmApp;

public class MainPageController
{
    @FXML private VBox topPanels;
    @FXML private TextField paramTextField1;
    @FXML private TextField paramTextField2;
    @FXML private TextField paramTextField3;
    @FXML private TextField paramTextField4;
    @FXML private TextField paramTextField5;
    @FXML private TextField leftBoundTextField;
    @FXML private TextField rightBoundTextField;
    @FXML private TextField pointsAmountTextField;
    @FXML private TextField populationSizeTextField;
    @FXML private TextField maxGenerationsTextField;
    @FXML private TextField aimedFitnessTextField;

    @FXML private Label dispersionLabel;
    @FXML private Slider dispersionSlider;

    @FXML private LineChart<Number, Number> chart;

    @FXML private Label algParametersLabel;
    @FXML private Label algInformationLabel;

    @FXML private Button btnSavePoints;
    @FXML private Button btnStop;
    @FXML private Button btnRun;

    /** Input function points */
    private float[] points;
    /** Input function points + gaussian noise */
    private float[][] noisedPoints;
    /** Input function points + gaussian noise; limited size for ui */
    private float[][] noisedPointsUI;
    /** Genetic algorithm results */
    private float[] calculatedPoints;

    private XYChart.Series<Number, Number> pointsSeries;
    private XYChart.Series<Number, Number> noisedPointsSeries;
    private XYChart.Series<Number, Number> calculatedPointsSeries;

    private GeneticAlgorithm geneticAlgorithm;
    private Thread thread;

    @FXML
    private void initialize()
    {
        initTextField(paramTextField1, "-10");
        initTextField(paramTextField2, "-8");
        initTextField(paramTextField3, "3");
        initTextField(paramTextField4, "1");
        initTextField(leftBoundTextField, "-4");
        initTextField(rightBoundTextField, "3");
        initTextField(pointsAmountTextField, "500");
        initTextField(populationSizeTextField, "1000");
        initTextField(maxGenerationsTextField, "5000");
        initTextField(aimedFitnessTextField, "0.0");

        geneticAlgorithm = new GeneticAlgorithm((params, isRunning, fitness, generation) ->
                Platform.runLater(() ->
                {
                    if (!isRunning)
                    {
                        stopAction();
                    }
                    algParametersLabel.setText("y = " + params[4] + "x⁴ + " + params[3] + "x³ + " +
                            params[2] + "x² + " + params[1] + "x + " + params[0]);
                    algInformationLabel.setText("Fintess: " + fitness + "  Generation: " + generation);
                    updateCalculatedPoints(params);
                })
        );

        dispersionSlider.valueProperty().addListener((changed, oldValue, newValue) -> {
            dispersionLabel.setText("Dispersion: " + ((int)(newValue.doubleValue()*10))/10.);
            updatePoints();
        });
        updatePoints();

        btnSavePoints.setOnAction(event -> directoryChooser());
        btnStop.setOnAction(event -> {
            stopAction();
            thread.interrupt();
        });
        btnRun.setOnAction(event -> runAction());
    }

    private void runAction()
    {
        btnStop.setDisable(false);
        btnRun.setDisable(true);
        topPanels.setDisable(true);

        int populationSize = getIntParam(populationSizeTextField);
        int maxGenerations = getIntParam(maxGenerationsTextField);
        float aimedFitness = getFloatParam(aimedFitnessTextField);
        thread = new Thread(() ->
        {
            try
            {
                geneticAlgorithm.execute(noisedPoints, populationSize, maxGenerations, aimedFitness);
            }
            catch (InterruptedException ignored) {}
            catch (Exception e)
            {
                stopAction();
            }
        });
        thread.setDaemon(true);
        thread.start();
    }

    private void stopAction()
    {
        btnStop.setDisable(true);
        btnRun.setDisable(false);
        topPanels.setDisable(false);
        btnRun.setText("Start");
    }

    @FXML
    public void updatePoints()
    {
        try
        {
            float param1 = getFloatParam(paramTextField1);
            float param2 = getFloatParam(paramTextField2);
            float param3 = getFloatParam(paramTextField3);
            float param4 = getFloatParam(paramTextField4);
            float param5 = getFloatParam(paramTextField5);
            float leftBound = getFloatParam(leftBoundTextField);
            float rightBound = getFloatParam(rightBoundTextField);
            int pointsAmount = getIntParam(pointsAmountTextField);
            double dispersion = dispersionSlider.getValue();

            float dx = (rightBound - leftBound) / (pointsAmount - 1);

            Random random = new Random(System.currentTimeMillis());
            points = new float[pointsAmount];
            noisedPoints = new float[2][pointsAmount];
            for (int i = 0; i < pointsAmount; i++)
            {
                float x = leftBound + dx * i;
                float y = (((param5 * x + param4) * x + param3) * x + param2) * x + param1;
                points[i] = y;
                noisedPoints[0][i] = x;
                noisedPoints[1][i] = (float) (y + random.nextGaussian() * dispersion);
            }

            updateChart();
        }
        catch (NumberFormatException | NegativeArraySizeException ignored)
        {}
    }

    public void updateCalculatedPoints(float[] params)
    {
        int pointsAmount = getIntParam(pointsAmountTextField);

        calculatedPoints = new float[pointsAmount];
        for (int i = 0; i < pointsAmount; i++)
        {
            float x = noisedPoints[0][i];
            float y = params[params.length-1];
            for (int k = params.length-2; k >= 0 ; k--)
            {
                y = (y*x + params[k]);
            }
            calculatedPoints[i] = y;
        }

        updateChart();
    }

    public void updateChart()
    {
        int pointsAmount = getIntParam(pointsAmountTextField);
        float leftBound = getFloatParam(leftBoundTextField);
        float rightBound = getFloatParam(rightBoundTextField);
        NumberAxis xAxis = (NumberAxis) chart.getXAxis();
        xAxis.setAutoRanging(false);
        xAxis.setLowerBound(leftBound);
        xAxis.setUpperBound(rightBound);

        ObservableList<XYChart.Series<Number, Number>> chartData = chart.getData();
        chartData.clear();

        pointsSeries = new XYChart.Series<>();
        noisedPointsSeries = new XYChart.Series<>();
        calculatedPointsSeries = new XYChart.Series<>();
        pointsSeries.setName("Function");
        noisedPointsSeries.setName("Noised points");
        calculatedPointsSeries.setName("Genetic algorithm result");
        chartData.add(pointsSeries);
        chartData.add(noisedPointsSeries);
        ObservableList<XYChart.Data<Number, Number>> pointsSeriesData = pointsSeries.getData();
        ObservableList<XYChart.Data<Number, Number>> noisedPointsSeriesSeriesData = noisedPointsSeries.getData();
        ObservableList<XYChart.Data<Number, Number>> calculatedPointsSeriesData = null;
        if (calculatedPoints != null)
        {
            chartData.add(calculatedPointsSeries);
            calculatedPointsSeriesData = calculatedPointsSeries.getData();
        }

        chart.setCreateSymbols(true);
        chart.getStylesheets().add(getClass().getResource("/chart.css").toExternalForm());
        final int UI_POINTS_CNT_LIMIT = (int) (GeneticAlgorithmApp.getPrimaryStage().getWidth()/4);
        double delta = (double) pointsAmount / Math.min(pointsAmount, UI_POINTS_CNT_LIMIT);
        for (float i = 0; i < pointsAmount; i += delta)
        {
            int ii = Math.min(Math.round(i), pointsAmount-1);
            float x = noisedPoints[0][ii];
            pointsSeriesData.add(new XYChart.Data<>(x, points[ii]));
            noisedPointsSeriesSeriesData.add(new XYChart.Data<>(x, noisedPoints[1][ii]));
            if (calculatedPoints != null)
            {
                calculatedPointsSeriesData.add(new XYChart.Data<>(x, calculatedPoints[ii]));
            }
        }

    }
    
    private int getIntParam(TextField textField)
    {
        String data = textField.getText();
        if (data == null || data.isEmpty())
        {
            data = textField.getPromptText();
        }
        return data == null || data.isEmpty() ? 0 : Integer.parseInt(data);
    }

    private float getFloatParam(TextField textField)
    {
        String data = textField.getText();
        if (data == null || data.isEmpty())
        {
            data = textField.getPromptText();
        }
        textField.setBorder(new Border(new BorderStroke(null, null, null, BorderWidths.FULL)));
        return data == null || data.isEmpty() ? 0.0f : Float.parseFloat(data);
    }

    private void initTextField(TextField textField, String value)
    {
        textField.setText(value);
        textField.setPromptText(value);
    }

    public void directoryChooser()
    {
        FileChooser chooser = new FileChooser();
        chooser.setInitialFileName("points.txt");
        chooser.setInitialDirectory(new File(System.getProperty("user.dir")));
        FileChooser.ExtensionFilter extensionFilter =
                new FileChooser.ExtensionFilter("TXT", "*.txt");
        chooser.getExtensionFilters().addAll(extensionFilter);
        chooser.setSelectedExtensionFilter(extensionFilter);

        File selectedFile = chooser.showSaveDialog(GeneticAlgorithmApp.getPrimaryStage());
        if (selectedFile == null)
        {
            return;
        }

        try (PrintWriter outFile = new PrintWriter(selectedFile))
        {
            outFile.write(Arrays.deepToString(noisedPoints));
        }
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }
    }

    @FXML
    public void onInputUpdate(ObservableValue observable, String oldValue, String newValue)
    {
        calculatedPoints = null;
        updatePoints();
    }
}
