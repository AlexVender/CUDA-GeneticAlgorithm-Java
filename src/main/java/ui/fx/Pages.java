package ui.fx;

public enum Pages
{
    main("", "page/MainPage.fxml");

    private String name;
    private String resource;

    Pages(final String name, final String resource)
    {
        this.name = name;
        this.resource = resource;
    }

    public String getName()
    {
        return name;
    }

    public String getResource()
    {
        return resource;
    }
}
