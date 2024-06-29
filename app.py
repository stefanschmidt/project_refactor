from flask import Flask, render_template
import clustering_label

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '<a href="clustering">Clustering</a>'

@app.route('/clustering')
def show_clustering():
    pca_plot_png_path = clustering_label.clustering()
    return render_template('clustering.html', pca_plot_png_path=pca_plot_png_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

