from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import pandas as pd
import os
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

HISTORY_FILE = 'history.csv'

# --------------------------
# Steganography Detector
# --------------------------


def analyze_image(image_path):
    """
    LSB Chi-Square analysis
    """
    img = Image.open(image_path).convert('L')
    pixels = np.array(img).flatten()

    # Histogram for chart
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    hist_data = hist.tolist()

    # Chi-Square LSB detection
    chi_score = 0.0
    for i in range(0, 256, 2):
        o_even = hist[i]
        o_odd = hist[i+1]
        n = o_even + o_odd
        if n == 0:
            continue
        e = n / 2
        chi_score += ((o_even - e)**2) / e + ((o_odd - e)**2) / e

    # Suspicion level
    if chi_score < 1000:
        suspicion_level = "Low"
    elif chi_score < 5000:
        suspicion_level = "Medium"
    else:
        suspicion_level = "High"

    return {
        "chi_score": round(float(chi_score), 2),
        "suspicion_level": suspicion_level,
        "hist_data": hist_data
    }

# --------------------------
# Save history
# --------------------------


def save_history(timestamp, filename, chi_score, suspicion_level):
    df = pd.DataFrame()
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
    # Append new row using pd.concat instead of deprecated append
    new_row = pd.DataFrame([{
        "Timestamp": timestamp,
        "Filename": filename,
        "Chi-Square": chi_score,
        "Suspicion Level": suspicion_level
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

# --------------------------
# Routes
# --------------------------


@app.route("/", methods=['GET', 'POST'])
def index():
    filename = None
    results = None
    records = []

    if os.path.exists(HISTORY_FILE):
        records = pd.read_csv(HISTORY_FILE).to_dict(orient='records')

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            results = analyze_image(filepath)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_history(timestamp, filename,
                         results['chi_score'], results['suspicion_level'])

            # Update records after saving
            records = pd.read_csv(HISTORY_FILE).to_dict(orient='records')

    return render_template("index.html", filename=filename, results=results, records=records)

# --------------------------
# Download PDF
# --------------------------


@app.route('/download_history')
def download_history():
    if not os.path.exists(HISTORY_FILE):
        return "No history to download.", 404

    df = pd.read_csv(HISTORY_FILE)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    data = [df.columns.tolist()] + df.values.tolist()

    # Automatic column width based on content length
    col_widths = []
    for col in df.columns:
        max_len = max(df[col].astype(str).map(len).max(), len(col))
        col_widths.append(max_len * 7)  # 7 px per character approx

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    doc.build([table])
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="history.pdf", mimetype='application/pdf')

# --------------------------
# Clear history
# --------------------------


@app.route("/clear_history")
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return redirect(url_for('index'))


# --------------------------
# Run Flask
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
