package ui.fx;

import java.io.IOException;
import java.net.URL;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class GeneticAlgorithmApp extends Application
{
    public static final String APP_TITLE = "Genetic Algorithm";

    private static GeneticAlgorithmApp geneticAlgorithmApp;
    private static Stage primaryStage;
    private static BorderPane root;

    @Override
    public void start(Stage primaryStage) throws Exception
    {
        GeneticAlgorithmApp.primaryStage = primaryStage;
        geneticAlgorithmApp = this;
        root = loadLayout("MenuLayout.fxml");

        primaryStage.setMinHeight(600);
        primaryStage.setMinWidth(800);
        primaryStage.setScene(new Scene(root));
        primaryStage.show();

        showPage(Pages.main);
    }

    public static <T extends Node> T loadLayout(String resourceName) throws IOException
    {
        URL resource = GeneticAlgorithmApp.class.getResource("/" + resourceName);
        FXMLLoader loader = new FXMLLoader(resource);
        T result = loader.load();
        result.getProperties().put("loader", loader);
        return result;
    }

    public static void showPage(Pages page)
    {
        showPage(page.getName(), page.getResource());
    }

    public static void showPage(String title, String resourceName)
    {
        try
        {
            updateTitle(title);
            Node node = GeneticAlgorithmApp.loadLayout(resourceName);
            root.setCenter(node);
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public static void updateTitle(String title)
    {
        if (title != null && !title.isEmpty())
        {
            primaryStage.setTitle(APP_TITLE + " – " + title);
        }
        else
        {
            primaryStage.setTitle(APP_TITLE);
        }
    }

    public static GeneticAlgorithmApp getGeneticAlgorithmApp()
    {
        return geneticAlgorithmApp;
    }

    public static Stage getPrimaryStage()
    {
        return primaryStage;
    }

    public static BorderPane getRoot()
    {
        return root;
    }

    public static void main(String[] args)
    {
        launch(args);
    }
}
