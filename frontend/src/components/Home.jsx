import React, { useEffect } from 'react';
import logo from '../assets/logo.svg';
import { useNavigate } from 'react-router-dom';

const Home = () => {
    const navigate = useNavigate();

    useEffect(() => {
        fetch('http://127.0.0.1:8000/api/sample/')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => console.log(data))
            .catch(error => console.error('Error fetching data:', error));
    }, []);

    const handleClick = () => {
        navigate('/chatbot');
    }

    return (
        <div className='flex flex-col items-center justify-center min-h-screen p-40'>
            <div className="shadow-card bg-primary p-6 rounded-lg">
                <div className='mb-6 flex justify-center items-center'>
                    <img src={logo} alt="logo" className='h-20' />
                </div>
                <div className='text-center'>
                    <h1 className='text-4xl font-bold mb-4'>Precisa de Ajuda?</h1>
                    {/* <p className='mb-8'>Selecione a opção que melhor lhe atende</p> */}
                    <div className='flex justify-center space-x-4'>
                        <button className='px-6 py-3 bg-yellow-400 text-black rounded-lg font-bold hover:bg-yellow-300 transition'
                        onClick={handleClick}
                        >Fale com o Hecate</button>
                        {/* <button className='px-6 py-3 bg-yellow-400 text-black rounded-lg font-bold hover:bg-yellow-300 transition'
                        onClick={handleClick}
                        >
                            Dúvidas sobre o mercado
                        </button> */}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Home;
