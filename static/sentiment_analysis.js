document.addEventListener('DOMContentLoaded', () => {

    document.querySelector('#live_tweets').onclick = () => {
         document.querySelector('#search_list').innerHTML="";
         document.querySelector("#donutchart").innerHTML="";
        // Initialize new request
        const request = new XMLHttpRequest();
        const search_query = document.querySelector('#form-search').value;
        const tweet_count = document.querySelector("#form-count").value;
        request.open('POST', '/live_tweets');

        if(search_query.trim()==="" || tweet_count.trim()===""){
            window.alert("Some of the field is missing !!!");
        }
        else{
            // Add data to send with request
            const data = new FormData();
            data.append('search_query', search_query);
            data.append('tweet_count',tweet_count);

            // Send request
            request.send(data);
        }
        // Callback function for when request completes
        request.onload = () => {

            // Extract JSON data from request
            const data = JSON.parse(request.responseText);

            // Update the result div
            if (data.success) {
                google.charts.load("current", {packages:["corechart"]});
                const d = google.visualization.arrayToDataTable([
                        ['Sentiment', 'Number'],
                        ['Negative', data.count[0]],
                        ['Positive', data.count[1]],
                        ['Neutral', data.count[2]]
                    ]);

                const options = {
                    title: "Sentiment Wheeel",
                    pieHole : 0.4,
                }

                const chart = new google.visualization.PieChart(document.getElementById('donutchart'));
                chart.draw(d, options);

                const thead = document.createElement('thead');
                const tbody = document.createElement('tbody');
                const th = document.createElement('th');
                const th1 = document.createElement('th');
                const th2 = document.createElement('th');
                const tr = document.createElement('tr');

                th.innerHTML = '#';
                tr.append(th);
                th1.innerHTML = 'Tweet';
                tr.append(th1);
                th2.innerHTML = 'Sentiment'
                tr.append(th2);

                thead.append(tr);
                document.querySelector('#search_list').append(thead);

                for(var i = 0; i<data.tweets.length ; i++){
                    /*const li = document.createElement('li');
                    const p = document.createElement('p');
                    p.innerHTML = data.tweets[i][0] + " : " + data.tweets[i][1];
                    li.append(p);
                    document.querySelector('#search_list').append(li);
                    */

                    const tr = document.createElement('tr');
                    const td = document.createElement('td');
                    const td1 = document.createElement('td');
                    const td2 = document.createElement('td');

                    td.innerHTML = i;
                    tr.append(td);
                    td1.innerHTML = data.tweets[i][0];
                    tr.append(td1);
                    td2.innerHTML = data.tweets[i][1];
                    tr.append(td2);

                    tbody.append(tr);
                }
                document.querySelector('#search_list').append(tbody);
            }
            else {
                window.alert("Check your Internet Connection !!!")
            }
        }

        return false;
    };

    document.querySelector('#offline_tweets').onclick = () => {
         document.querySelector('#search_list').innerHTML="";
         document.querySelector("#donutchart").innerHTML="";
        // Initialize new request
        const request = new XMLHttpRequest();
        const search_query = document.querySelector('#form-search').value;
        const tweet_count = document.querySelector("#form-count").value;
        request.open('POST', '/offline_tweets');

        if(search_query.trim()==="" || tweet_count.trim()===""){
            window.alert("Some of the field is missing !!!");
        }
        else{
            const data = new FormData();
            data.append('search_query', search_query);
            data.append('tweet_count',tweet_count);

            // Send request
            request.send(data);
        }
        
        // Callback function for when request completes
        request.onload = () => {

            // Extract JSON data from request
            const data = JSON.parse(request.responseText);

            // Update the result div
            if (data.success) {
                google.charts.load("current", {packages:["corechart"]});
                const d = google.visualization.arrayToDataTable([
                        ['Sentiment', 'Number'],
                        ['Negative', data.count[0]],
                        ['Positive', data.count[1]],
                        ['Neutral', data.count[2]]
                    ]);

                const options = {
                    title: "Sentiment Wheeel",
                    pieHole : 0.4,
                }

                const chart = new google.visualization.PieChart(document.getElementById('donutchart'));
                chart.draw(d, options);

                const thead = document.createElement('thead');
                const tbody = document.createElement('tbody');
                const th = document.createElement('th');
                const th1 = document.createElement('th');
                const th2 = document.createElement('th');
                const tr = document.createElement('tr');

                th.innerHTML = '#';
                tr.append(th);
                th1.innerHTML = 'Tweet';
                tr.append(th1);
                th2.innerHTML = 'Sentiment'
                tr.append(th2);

                thead.append(tr);
                document.querySelector('#search_list').append(thead);

                for(var i = 0; i<data.tweets.length ; i++){
                    const tr = document.createElement('tr');
                    const td = document.createElement('td');
                    const td1 = document.createElement('td');
                    const td2 = document.createElement('td');

                    td.innerHTML = i;
                    tr.append(td);
                    td1.innerHTML = data.tweets[i][0];
                    tr.append(td1);
                    td2.innerHTML = data.tweets[i][1];
                    tr.append(td2);

                    tbody.append(tr);
                }
                document.querySelector('#search_list').append(tbody);
            }
            else {
                window.alert("No Tweet Found !!!");
            }
        }

        return false;
    };

});