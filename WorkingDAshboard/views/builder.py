# views/builder.py — Level 2: custom graph builder
import data
import graph_builder as gb
import ui

df, *_ = data.get_data()

ui.page_header("Level 2 · Graph Builder", "Build your own chart",
               "Choose what you want to see, narrow down the data, and fine-tune the view. "
               "The chart updates instantly.")
gb.render_graph_builder(df)
ui.footer()
