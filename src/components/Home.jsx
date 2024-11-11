import React, { useEffect } from 'react'


const Home = () => {
    useEffect(() => {
        
        fetch('/api/sample/')
            .then(response => response.json())
            .then(data => console.log(data));

    }, []);

    return (
        <div>
            <h1>Home Page</h1>
            <p>Welcome to the Home Page!</p>
        </div>
    )
}

export default Home;
