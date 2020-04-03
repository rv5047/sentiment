document.addEventListener('DOMContentLoaded', () => {

    document.querySelector('#search_btn').onclick = () => {
         document.querySelector('#search_list').innerHTML="";

        // Initialize new request
        const request = new XMLHttpRequest();
        const search_query = document.querySelector('#form-search').value;
        const tweet_count = document.querySelector("#form-count").value;
        request.open('POST', '/search');

        // Callback function for when request completes
        request.onload = () => {

            // Extract JSON data from request
            const data = JSON.parse(request.responseText);

            // Update the result div
            if (data.success) {
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
                // document.querySelector('#result').innerHTML = contents;
            }
            else {
                // document.querySelector('#result').innerHTML = 'There was an error.';
            }
        }

        // Add data to send with request
        const data = new FormData();
        data.append('search_query', search_query);
        data.append('tweet_count',tweet_count);

        // Send request
        request.send(data);
        return false;

    };


});