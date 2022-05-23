import './App.css';
import { BrowserRouter, Route, Switch } from 'react-router-dom'
import Home from './components/Home';
import Results from './components/results';

function App() {



  return (

    <BrowserRouter>
      <div className="App-mis">
        <Switch >
          <Route exact path='/' component={Home} />
          <Route path='/results' component={Results} />
          {/* <Route component={PageNotFound} /> */}
        </Switch>
      </div>
    </BrowserRouter>
  );
}

export default App;