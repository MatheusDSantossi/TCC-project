import React, { useEffect } from 'react';
import logo from '../assets/logo.svg';
import { useNavigate } from 'react-router-dom';
import AxiosInstance from './AxiosInstance';

const Home = () => {
    const navigate = useNavigate();

    const TestAPI = () => {
        AxiosInstance.get('http://127.0.0.1:8000/api/sample/')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => console.log(data))
            .catch(error => console.error('Error fetching data:', error));
    }

    useEffect(() => {
        TestAPI
    }, []);

    const handleClick = () => {
        navigate('/chatbot');
    }

    return (
        <div className='flex flex-col items-center justify-center min-h-screen p-40'>
            <div className="p-6 rounded-lg shadow-card bg-primary">
                <div className='flex items-center justify-center mb-6'>
                    <img src={logo} alt="logo" className='h-20' />
                </div>
                <div className='text-center'>
                    <h1 className='mb-4 text-4xl font-bold'>Precisa de Ajuda?</h1>
                    {/* <p className='mb-8'>Selecione a opção que melhor lhe atende</p> */}
                    <div className='flex justify-center space-x-4'>
                        <button className='px-6 py-3 font-bold text-black transition bg-yellow-400 rounded-lg hover:bg-yellow-300'
                        onClick={handleClick}
                        >Fale com o Hecate</button>
                        {/* <button className='px-6 py-3 font-bold text-black transition bg-yellow-400 rounded-lg hover:bg-yellow-300'
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
