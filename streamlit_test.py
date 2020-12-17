import streamlit as st
import streamlit.components.v1 as components

# test using streamlit instead since this seems to be a better data
# framework for apps
# doesn't work yet

# >>> import plotly.express as px
# >>> fig = px.box(range(10))
# >>> fig.write_html('test.html')

st.header("test html import")

HtmlFile = open("test_report.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
#print(source_code)
components.html(source_code)