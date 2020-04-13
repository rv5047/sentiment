document.addEventListener('DOMContentLoaded', () => {

    document.querySelector('#delete_tweets').onclick = () => {
        document.querySelector('#search_list').innerHTML="";

        const request = new XMLHttpRequest();

        if(window.confirm("Really want to delete tweets ?")){
            request.open('POST', '/delete');
            }
        else{

        }

        // Callback function for when request completes
        request.onload = () => {

            // Extract JSON data from request
            const data = JSON.parse(request.responseText);

            if (data.success) {
                window.alert("Deleted");
            }
            else {
            }
        }
        const data = new FormData();

        // Send request
        request.send(data);
        return false;

    };

    document.querySelector('#fetch_tweets').onclick = () => {
        document.querySelector('#search_list').innerHTML="";

        const request = new XMLHttpRequest();
        const search_query = document.querySelector('#form-search').value;
        request.open('POST', '/fetch_tweets');
        
        if(search_query.trim()===""){
            window.alert("Some of the field is missing !!!");
        }
        else{
            const data = new FormData();
            data.append('search_query', search_query);

            // Send request
            request.send(data);
        }

        request.onload = () => {

            const data = JSON.parse(request.responseText);

            if (data.success) {
                window.alert("Completed");
            }
            else {
                window.alert("Check you Internet Connection !!!")
            }
        }
      
        return false;
    };

    document.querySelector('#topical_modeling').onclick = () => {
        document.querySelector('#search_list').innerHTML="";

        const request = new XMLHttpRequest();
        request.open('POST', '/topical_modeling');

        request.onload = () => {

            const data = JSON.parse(request.responseText);

            if (data.success) {
                const thead = document.createElement('thead');
                const tbody = document.createElement('tbody');
                const th = document.createElement('th');
                const th1 = document.createElement('th');
                const th2 = document.createElement('th');
                const tr = document.createElement('tr');

                th.innerHTML = 'Topic #';
                tr.append(th);
                th1.innerHTML = 'Words';
                tr.append(th1);

                thead.append(tr);
                document.querySelector('#search_list').append(thead);

                for(var i = 0; i<data.topic.length ; i++){

                    const tr = document.createElement('tr');
                    const td = document.createElement('td');
                    const td1 = document.createElement('td');

                    td.innerHTML = i;
                    tr.append(td);
                    td1.innerHTML = data.topic[i];
                    tr.append(td1);
                    tbody.append(tr);
                }
                document.querySelector('#search_list').append(tbody);
            }
            else if (data.success == false){
                window.alert("File not found !!!")
            }
            else{
                window.alert("File size is less..\nPlease click Fetch Tweets button")
            }
        }

        const data = new FormData();
        request.send(data);        
        return false;

    };
});