document.addEventListener('DOMContentLoaded', () => {

    document.querySelector('#delete_tweets').onclick = () => {
        document.querySelector('#search_result').innerHTML="";

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
        document.querySelector('#search_result').innerHTML="";

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
        document.querySelector('#search_result').innerHTML="";

        const request = new XMLHttpRequest();
        request.open('POST', '/topical_modeling');

        request.onload = () => {

            const data = JSON.parse(request.responseText);

            const paths = [['../static/img/topic_clustering/cloud0.png', '../static/img/topic_clustering/bar0.png'], 
                           ['../static/img/topic_clustering/cloud1.png', '../static/img/topic_clustering/bar1.png'],
                           ['../static/img/topic_clustering/cloud2.png', '../static/img/topic_clustering/bar2.png'],
                           ['../static/img/topic_clustering/cloud3.png', '../static/img/topic_clustering/bar3.png'],
                           ['../static/img/topic_clustering/cloud4.png', '../static/img/topic_clustering/bar4.png']];

            if (data.success) {
                function setAttributes(el, attrs) {
                  for(var key in attrs) {
                    el.setAttribute(key, attrs[key]);
                  }
                }

                for(var i=0; i<5; i++){
                    const div = document.createElement('div');
                    const div1 = document.createElement('div');
                    const div2 = document.createElement('div');
                    const img = document.createElement('img');
                    const img1 = document.createElement('img');

                    div.classList.add('row');
                    div1.classList.add('col-sm-6');
                    div2.classList.add('col-sm-6');

                    div.setAttribute('id', 'image');
                    setAttributes(img,{'src': paths[i][0]});
                    setAttributes(img1,{'src':paths[i][1]});

                    div1.append(img);
                    div2.append(img1);
                    div.append(div1);
                    div.append(div2);
                    document.querySelector('#search_result').append(div);
                }
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