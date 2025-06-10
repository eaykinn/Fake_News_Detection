const algorithms = ['Naive Bayes', 'Logistic Regression', 'Decision Tree', 'KNN'];
const metrics = ['accuracy', 'precision', 'recall', 'f1-score'];


document.getElementById("algoritmaForm").addEventListener("submit", function(e) {
    e.preventDefault();
    spinner = document.getElementById("loading");
    spinner.style.display = "flex"; //
    spinner.style.alignItems = "center";

   
    const metin = document.getElementById("habermetni").value;


    const algoritmalar = [];
    document.querySelectorAll('#algoritmaForm input[type="checkbox"]:checked').forEach(cb => {
        algoritmalar.push(cb.value);
    });


    const formData = new URLSearchParams();
    formData.append("habermetni", metin);
    formData.append("algoritmalar", JSON.stringify(algoritmalar)); 

    fetch("/import-data", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: formData.toString()
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("sonuc").innerText = "Hata: " + data.error;
        } else {
           document.getElementById("nb_sonuc").innerHTML = getResultIcon(data.sonuc1);
            document.getElementById("lr_sonuc").innerHTML = getResultIcon(data.sonuc2);
            document.getElementById("dt_sonuc").innerHTML = getResultIcon(data.sonuc3);
            document.getElementById("knn_sonuc").innerHTML = getResultIcon(data.sonuc4);
            
        }
        getPerformances(algorithms)
    })
    .finally(() => {
        document.getElementById("loading").style.display = "none";
    })
    .catch(err => {
        document.getElementById("sonuc").innerText = "Hata: " + err;
    });

 
    
});


function getPerformances(algoritmalar) {
    const formData = new URLSearchParams();
    formData.append("algoritmalar", JSON.stringify(algoritmalar));
    fetch("/get-performances", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formData.toString()
    })
    .then(response => response.json())
    .then(data => {
        if (data["Naive Bayes"]) {
            console.log(data["Naive Bayes"]);
            document.getElementById("nb_accuracy").innerText = data["Naive Bayes"].accuracy.toFixed(3);
            document.getElementById("nb_precision").innerText = data["Naive Bayes"]["classification_report"]["weighted avg"].precision.toFixed(3);
            document.getElementById("nb_recall").innerText = data["Naive Bayes"]["classification_report"]["weighted avg"].recall.toFixed(3);
            document.getElementById("nb_f1score").innerText = data["Naive Bayes"]["classification_report"]["weighted avg"]["f1-score"].toFixed(3);
            document.getElementById("nb_support").innerText = data["Naive Bayes"]["classification_report"]["weighted avg"].support.toFixed(3);
            document.getElementById("nb_duration").innerText = data["Naive Bayes"].duration.toFixed(3) + "  sec";
            document.getElementById("nb_memory").innerText = data["Naive Bayes"]["memory_usage_bytes"].toFixed(3) + " MB";
            document.getElementById("nb_row").hidden = false;
            document.getElementById("nbConfusionMatrix").hidden = false;
        }else{
            document.getElementById("nb_row").hidden = true;
            document.getElementById("nbConfusionMatrix").hidden = true;
         
        }
        if (data["Logistic Regression"]) {
            document.getElementById("lr_accuracy").innerText = data["Logistic Regression"].accuracy.toFixed(3);
            document.getElementById("lr_precision").innerText = data["Logistic Regression"]["classification_report"]["weighted avg"].precision.toFixed(3);
            document.getElementById("lr_recall").innerText = data["Logistic Regression"]["classification_report"]["weighted avg"].recall.toFixed(3);
            document.getElementById("lr_f1score").innerText = data["Logistic Regression"]["classification_report"]["weighted avg"]["f1-score"].toFixed(3);
            document.getElementById("lr_support").innerText = data["Logistic Regression"]["classification_report"]["weighted avg"].support.toFixed(3);
            document.getElementById("lr_duration").innerText = data["Logistic Regression"].duration.toFixed(3)+ " sec";
            document.getElementById("lr_memory").innerText = data["Logistic Regression"]["memory_usage_bytes"].toFixed(3) + " MB";
            document.getElementById("lr_row").hidden = false;
            document.getElementById("lrConfusionMatrix").hidden = false;
        }else{
            document.getElementById("lr_row").hidden = true;
            document.getElementById("lrConfusionMatrix").hidden = true;
          
         
        }
        if (data["Decision Tree"]) {
            document.getElementById("dt_accuracy").innerText = data["Decision Tree"].accuracy.toFixed(3);
            document.getElementById("dt_precision").innerText = data["Decision Tree"]["classification_report"]["weighted avg"].precision.toFixed(3);
            document.getElementById("dt_recall").innerText = data["Decision Tree"]["classification_report"]["weighted avg"].recall.toFixed(3);
            document.getElementById("dt_f1score").innerText = data["Decision Tree"]["classification_report"]["weighted avg"]["f1-score"].toFixed(3);
            document.getElementById("dt_support").innerText = data["Decision Tree"]["classification_report"]["weighted avg"].support.toFixed(3);
            document.getElementById("dt_duration").innerText = data["Decision Tree"].duration.toFixed(3) + " sec";
            document.getElementById("dt_memory").innerText = data["Decision Tree"]["memory_usage_bytes"].toFixed(3) + " MB";
            document.getElementById("dt_row").hidden = false;
            document.getElementById("dtConfusionMatrix").hidden = false;
        }else{
            document.getElementById("dt_row").hidden = true;
            document.getElementById("dtConfusionMatrix").hidden = true;
           
            
        }
        if (data["KNN"]) {
            document.getElementById("knn_accuracy").innerText = data["KNN"].accuracy.toFixed(3);
            document.getElementById("knn_precision").innerText = data["KNN"]["classification_report"]["weighted avg"].precision.toFixed(3);
            document.getElementById("knn_recall").innerText = data["KNN"]["classification_report"]["weighted avg"].recall.toFixed(3);
            document.getElementById("knn_f1score").innerText = data["KNN"]["classification_report"]["weighted avg"]["f1-score"].toFixed(3);
            document.getElementById("knn_support").innerText = data["KNN"]["classification_report"]["weighted avg"].support.toFixed(3);
            document.getElementById("knn_duration").innerText = data["KNN"].duration.toFixed(3) + " sec";
            document.getElementById("knn_memory").innerText = data["KNN"]["memory_usage_bytes"].toFixed(3) + " MB";
            document.getElementById("knn_row").hidden = false;
            document.getElementById("knnConfusionMatrix").hidden = false;
        }else{
            document.getElementById("knn_row").hidden = true;
            document.getElementById("knnConfusionMatrix").hidden = true;
            
           
        }
        
        if (window.radarChart) window.radarChart.destroy();
        if (window.barChart) window.barChart.destroy();
        if (window.barChart2) window.barChart2.destroy();

        document.querySelector("#apexRadarChart").innerHTML = "";
        document.querySelector("#apexBarChart").innerHTML = "";
        document.querySelector("#apexBarChart2").innerHTML = "";


      
        const radarSeries = algoritmalar
        .filter(algo => data[algo])
        .map(algo => ({
            name: algo,
            data: [
                data[algo].accuracy || 0,
                data[algo]["classification_report"]["weighted avg"].precision || 0,
                data[algo]["classification_report"]["weighted avg"].recall || 0,
                data[algo]["classification_report"]["weighted avg"]["f1-score"] || 0
            ]
        }));

        const radarOptions = {
            chart: { type: 'radar', height: 350 },
            series: radarSeries,
            labels: metrics.map(m => m.charAt(0).toUpperCase() + m.slice(1)),
            title: { text: 'Algorithms Radar Chart' }
        };

        const radarChart = new ApexCharts(document.querySelector("#apexRadarChart"), radarOptions);
        radarChart.render();
        


        const barSeries = algoritmalar
        .filter(algo => data[algo])
        .map(algo => ({
            name: algo,
            data: [
                data[algo].duration.toFixed(3) || 0,
            ]
        }
        ));

        const barOptions = {
            chart: { type: 'bar', height: 350 },
            series: barSeries,
            xaxis: {
                categories: ['Run Time (sec)']
            },
            title: { text: 'Algorithms Run Time' }
        };

        const barChart = new ApexCharts(document.querySelector("#apexBarChart"), barOptions);
        barChart.render();




         const barSeries2 = algoritmalar
         .filter(algo => data[algo])
         .map(algo => ({
            name: algo,
            data: [
                data[algo].memory_usage_bytes.toFixed(3) || 0,
            ]
        }
        ));

        const barOptions2 = {
            chart: { type: 'bar', height: 350 },
            series: barSeries2,
            xaxis: {
                categories: ['Memory Usage (MB)']
            },
            title: { text: 'Algorithms Memory Usage' }
        };

        const barChart2 = new ApexCharts(document.querySelector("#apexBarChart2"), barOptions2);
        barChart2.render();

        if (data["Naive Bayes"]) {
            renderConfusionMatrix(
                data["Naive Bayes"]["confusion_matrix"],
                "nbConfusionMatrix",
                ["True", "Fake"]
            );
        }
        if (data["Decision Tree"]) {
            renderConfusionMatrix(
                data["Decision Tree"]["confusion_matrix"],
                "dtConfusionMatrix",
                ["True", "Fake"]
            );
        }
        if (data["Logistic Regression"]) {
            renderConfusionMatrix(
                data["Logistic Regression"]["confusion_matrix"],
                "lrConfusionMatrix",
                ["True", "Fake"]
            );
        }
        if (data["KNN"]) {
            renderConfusionMatrix(
                data["KNN"]["confusion_matrix"],
                "knnConfusionMatrix",
                ["True", "Fake"]
            );
        }
    });

    
}

getPerformances(algorithms)

function renderConfusionMatrix(matrix, containerId, labels=["True", "Fake"]) {
    let html = `
        <table class="table table-hover">
            <thead class="table-dark">
                <tr>
                    <th></th>
                    <th>${labels[0]}</th>
                    <th>${labels[1]}</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <th >${labels[0]}</th>
                    <td>${matrix[0][0]}</td>
                    <td>${matrix[0][1]}</td>
                </tr>
                <tr>
                    <th >${labels[1]}</th>
                    <td>${matrix[1][0]}</td>
                    <td>${matrix[1][1]}</td>
                </tr>
            </tbody>
        </table>
    `;
    document.getElementById(containerId).innerHTML = html;
}


function getResultIcon(result) {
    if (result === true || result === "True") {
        
        return '<span style="color:green;font-size:1.5em;">&#10004;</span> True';
    } else if (result === false || result === "False") {
        
        return '<span style="color:red;font-size:1.5em;">&#10008;</span> False';
    } else {
        return '';
    }
}

function clearSelectedModels(algoritmalar) {
    const formData = new URLSearchParams();
    formData.append("algoritmalar", JSON.stringify(algoritmalar));
    fetch("/clear-models", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formData.toString()
    })
    .then(response => response.json())
    .then(data => {
        alert("Temizlenen modeller: " + data.cleared.join(", "));
        getPerformances(algorithms);
    });
}

function getSelectedAlgorithms() {
    const algoritmalar = [];
    document.querySelectorAll('#algoritmaForm input[type="checkbox"]:checked').forEach(cb => {
        algoritmalar.push(cb.value);
    });
    return algoritmalar;
}

        // Tab aktivasyonu
        var tabElms = document.querySelectorAll('a[data-bs-toggle="tab"]');
        tabElms.forEach(function(tabElm) {
            tabElm.addEventListener('shown.bs.tab', function (event) {
                event.target // newly activated tab
                event.relatedTarget // previous active tab
            });
        });