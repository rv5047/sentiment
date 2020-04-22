document.addEventListener('DOMContentLoaded', () => {

    document.querySelector('#live_tweets').onclick = () => {
         document.querySelector('#search_list').innerHTML="";

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
                function setAttributes(el, attrs) {
                  for(var key in attrs) {
                    el.setAttribute(key, attrs[key]);
                  }
                }

                const div = document.createElement('div');
                const div1 = document.createElement('div');
                const div2 = document.createElement('div');
                const img = document.createElement('img');
                const img1 = document.createElement('img');

                div.classList.add('row');
                div1.classList.add('col-sm-6');
                div2.classList.add('col-sm-6');

                div.setAttribute('id', 'image');
                setAttributes(img,{'src': "./static/img/sentiment_analysis/pie.png", 'width' : '500', 'height' : '500'});
                setAttributes(img1,{'src': "./static/img/sentiment_analysis/cloud.png", 'width' : '500', 'height' : '500'});

                div1.append(img);
                div2.append(img1);
                div.append(div1);
                div.append(div2);
                document.querySelector('#search_result').append(div);
                
                const table = document.createElement('table');
                tabel.setAttribute('id', 'search_list');
                table.classList.add('table');
                document.querySelector('#search_result').append(tabel);                

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
                window.alert("Check your Internet Connection !!!")
            }
        }

        return false;
    };

    document.querySelector('#offline_tweets').onclick = () => {
         document.querySelector('#search_list').innerHTML="";

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
                function setAttributes(el, attrs) {
                  for(var key in attrs) {
                    el.setAttribute(key, attrs[key]);
                  }
                }

                const div = document.createElement('div');
                const div1 = document.createElement('div');
                const div2 = document.createElement('div');
                const img = document.createElement('img');
                const img1 = document.createElement('img');

                div.classList.add('row');
                div1.classList.add('col-sm-6');
                div2.classList.add('col-sm-6');

                div.setAttribute('id', 'image');
                setAttributes(img,{'src': "./static/img/sentiment_analysis/pie.png", 'width' : '500', 'height' : '500'});
                setAttributes(img1,{'src': "./static/img/sentiment_analysis/cloud.png", 'width' : '500', 'height' : '500'});

                div1.append(img);
                div2.append(img1);
                div.append(div1);
                div.append(div2);
                document.querySelector('#search_result').append(div);
                
                const table = document.createElement('table');
                tabel.setAttribute('id', 'search_list');
                table.classList.add('table');
                document.querySelector('#search_result').append(tabel);                

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