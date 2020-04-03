
document.addEventListener('DOMContentLoaded', () => {

    document.querySelector('#form').onsubmit = () => {
         document.querySelector('#search_list').innerHTML="";

        // Initialize new request
        const request = new XMLHttpRequest();
        const search_query = document.querySelector('#form-username').value;
        request.open('POST', '/search');

        // Callback function for when request completes
        request.onload = () => {

            // Extract JSON data from request
            const data = JSON.parse(request.responseText);

            // Update the result div
            if (data.success) {

                for(var i = 0; i<data.tweets.length ; i++){
                    const li = document.createElement('li');
                    const p = document.createElement('p');
                    // li.innerHTML = data.tweets[i][0];
                    p.innerHTML = data.tweets[i][0] + " : " + data.tweets[i][1];
                    li.append(p);
                    document.querySelector('#search_list').append(li);



                }
                // document.querySelector('#result').innerHTML = contents;
            }
            else {
                // document.querySelector('#result').innerHTML = 'There was an error.';
            }
        }

        // Add data to send with request
        const data = new FormData();
        data.append('search_query', search_query);

        // Send request
        request.send(data);
        return false;

    };


});