package ui.fx.controller;

import javafx.fxml.FXML;
import ui.fx.GeneticAlgorithmApp;


public class MenuLayoutController
{
    
    @FXML
    private void handleAccounts() {
        GeneticAlgorithmApp.showPage("Счета", "page/AccountPage.fxml");
    }
    
    @FXML
    private void handleIncomeAndExpenses() {
        GeneticAlgorithmApp.showPage("Доходы и расходы", "page/IncomeAndExpensesPage.fxml");
    }
    
    @FXML
    private void handleOperations() {
        GeneticAlgorithmApp.showPage("Операции", "page/OperationsPage.fxml");
    }
    
}
