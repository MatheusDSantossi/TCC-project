import logo from '../../assets/logo.svg';
import { useNavigate } from 'react-router-dom';

const AdminHome = () => {
    const navigate = useNavigate();

    const handleClick = (e) => {
        
        navigate("/register");
    }

    return (
        <div className='flex flex-col items-center justify-center min-h-screen p-40'>
            <div className="p-6 rounded-lg shadow-card bg-primary">
                <div className='flex items-center justify-center mb-6'>
                    <img src={logo} alt="logo" className='h-20' />
                </div>
                <div className='text-center'>
                    <h1 className='mb-4 text-4xl font-bold'>Ferramentas administrativas</h1>
                    {/* <p className='mb-8'>Selecione a opção que melhor lhe atende</p> */}
                    <div className='flex justify-center space-x-4'>
                        <button className='px-6 py-3 font-bold text-black transition bg-yellow-400 rounded-lg hover:bg-yellow-300'
                        onClick={(e) => handleClick(e)}
                        >Login</button>
                        <button className='px-6 py-3 font-bold text-black transition bg-yellow-400 rounded-lg hover:bg-yellow-300'
                        onClick={(e) => handleClick(e)}
                        >
                            Signup
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default AdminHome;
