import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.graph_objs as go
import numpy as np
from scipy import stats
import math

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Dictionary containing distribution formulas
distribution_formulas = {
    'Uniform': r'f(x) = \frac{1}{b-a} \text{ for } a \leq x \leq b',
    'Exponential': r'f(x) = \lambda e^{-\lambda x} \text{ for } x \geq 0',
    'Normal': r'f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}',
    'Poisson': r'P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}',
    'Geometric': r'P(X = k) = p(1-p)^{k-1}',
    'Binomial': r'P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}',
    'Negative Binomial': r'P(X = k) = \binom{k+r-1}{k}p^r(1-p)^k',
    'Gamma': r'f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}',
    'Beta': r'f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}',
    'Chi-square': r'f(x) = \frac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}e^{-x/2}',
    'Pareto': r'f(x) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}',
    'Student\'s T': r'f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})}(1+\frac{x^2}{\nu})^{-\frac{\nu+1}{2}}',
    'Weibull': r'f(x) = \frac{k}{\lambda}(\frac{x}{\lambda})^{k-1}e^{-(x/\lambda)^k}'
}

# Short descriptions and use-cases for each distribution
distribution_descriptions = {
    'Uniform': "Every outcome in the range [a, b] is equally likely. Used when there's no reason to favor any value over another.",
    'Exponential': "Models the time between independent events occurring at a constant rate. Memoryless: the future is independent of the past.",
    'Normal': "The classic bell curve. Arises naturally when many independent factors add together (Central Limit Theorem).",
    'Poisson': "Counts the number of events occurring in a fixed interval when events happen at a constant average rate independently.",
    'Geometric': "Counts the number of trials until the first success. Used in reliability and queuing theory.",
    'Binomial': "Counts successes in n independent yes/no trials each with probability p. Foundation of A/B testing.",
    'Negative Binomial': "Counts failures before the r-th success. Used in overdispersed count data (e.g., RNA-seq).",
    'Gamma': "Generalizes the Exponential to model waiting time until the α-th event. Used in insurance and hydrology.",
    'Beta': "Models a probability or proportion — values constrained to [0,1]. Used as a Bayesian prior for probability estimates.",
    'Chi-square': "Sum of squared standard normals. Backbone of hypothesis tests for categorical data and variance estimation.",
    'Pareto': "Models \"80/20\" power-law phenomena — a small fraction causes most of the effect. Used for wealth, city sizes, web traffic.",
    'Student\'s T': "Like the Normal but with heavier tails, especially for small samples. Used when population variance is unknown.",
    'Weibull': "Flexible model for time-to-failure. Shape parameter k < 1: infant mortality; k = 1: random failure; k > 1: wear-out."
}

# Real-world scenario presets for each distribution
distribution_scenarios = {
    'Uniform': [
        {'label': 'Random number (0–1)', 'params': {'a': 0, 'b': 1}},
        {'label': 'Bus arrival (0–10 min wait)', 'params': {'a': 0, 'b': 10}},
        {'label': 'Dice roll (1–6)', 'params': {'a': 1, 'b': 6}},
    ],
    'Exponential': [
        {'label': 'Server request (λ=2/sec)', 'params': {'lambda': 2}},
        {'label': 'Radioactive decay (λ=0.1)', 'params': {'lambda': 0.1}},
        {'label': 'Call center (avg 5 min, λ=0.2)', 'params': {'lambda': 0.2}},
    ],
    'Normal': [
        {'label': 'Adult height (μ=170cm, σ=10)', 'params': {'mu': 170, 'sigma': 10}},
        {'label': 'IQ scores (μ=100, σ=15)', 'params': {'mu': 100, 'sigma': 15}},
        {'label': 'Manufacturing tolerance (μ=0, σ=0.5)', 'params': {'mu': 0, 'sigma': 0.5}},
    ],
    'Poisson': [
        {'label': 'Website hits/min (λ=5)', 'params': {'lambda': 5}},
        {'label': 'Defects per unit (λ=1.5)', 'params': {'lambda': 1.5}},
        {'label': 'Emails per hour (λ=20)', 'params': {'lambda': 20}},
    ],
    'Geometric': [
        {'label': 'Fair coin (p=0.5)', 'params': {'p': 0.5}},
        {'label': 'Sales conversion (p=0.1)', 'params': {'p': 0.1}},
        {'label': 'Free throw (p=0.75)', 'params': {'p': 0.75}},
    ],
    'Binomial': [
        {'label': 'A/B test (n=100, p=0.5)', 'params': {'n': 100, 'p': 0.5}},
        {'label': 'Drug trial (n=50, p=0.3)', 'params': {'n': 50, 'p': 0.3}},
        {'label': 'Quality control (n=20, p=0.95)', 'params': {'n': 20, 'p': 0.95}},
    ],
    'Negative Binomial': [
        {'label': 'Sales calls until 5 deals (p=0.2)', 'params': {'r': 5, 'p': 0.2}},
        {'label': 'Games until 3 wins (p=0.6)', 'params': {'r': 3, 'p': 0.6}},
    ],
    'Gamma': [
        {'label': 'Insurance claims (α=2, β=0.5)', 'params': {'alpha': 2, 'beta': 0.5}},
        {'label': 'Rainfall amount (α=5, β=1)', 'params': {'alpha': 5, 'beta': 1}},
        {'label': 'Server load (α=3, β=2)', 'params': {'alpha': 3, 'beta': 2}},
    ],
    'Beta': [
        {'label': 'Uninformative prior (α=1, β=1)', 'params': {'alpha': 1, 'beta': 1}},
        {'label': 'Conversion rate (α=5, β=45)', 'params': {'alpha': 5, 'beta': 45}},
        {'label': 'Task completion % (α=8, β=2)', 'params': {'alpha': 8, 'beta': 2}},
    ],
    'Chi-square': [
        {'label': 'Goodness-of-fit (k=3)', 'params': {'df': 3}},
        {'label': 'Variance test (k=10)', 'params': {'df': 10}},
        {'label': 'Large sample (k=30)', 'params': {'df': 30}},
    ],
    'Pareto': [
        {'label': 'Wealth (80/20 rule, α=1.16)', 'params': {'alpha': 1.16, 'xm': 1}},
        {'label': 'City population (α=2)', 'params': {'alpha': 2, 'xm': 10000}},
        {'label': 'File sizes (α=3)', 'params': {'alpha': 3, 'xm': 1}},
    ],
    'Student\'s T': [
        {'label': 'Small sample (ν=3)', 'params': {'df': 3}},
        {'label': 'Moderate sample (ν=10)', 'params': {'df': 10}},
        {'label': 'Near-normal (ν=30)', 'params': {'df': 30}},
    ],
    'Weibull': [
        {'label': 'Infant mortality (k=0.5)', 'params': {'k': 0.5, 'lambda': 1}},
        {'label': 'Random failure (k=1, Exponential)', 'params': {'k': 1, 'lambda': 1}},
        {'label': 'Wear-out failure (k=3)', 'params': {'k': 3, 'lambda': 2}},
    ],
}

# Dictionary mapping distributions to their parameter inputs
distribution_params = {
    'Uniform': [
        {'name': 'a (minimum)', 'id': 'a', 'default': 0},
        {'name': 'b (maximum)', 'id': 'b', 'default': 1}
    ],
    'Exponential': [
        {'name': 'λ (rate)', 'id': 'lambda', 'default': 1}
    ],
    'Normal': [
        {'name': 'μ (mean)', 'id': 'mu', 'default': 0},
        {'name': 'σ (std dev)', 'id': 'sigma', 'default': 1}
    ],
    'Poisson': [
        {'name': 'λ (rate)', 'id': 'lambda', 'default': 1}
    ],
    'Geometric': [
        {'name': 'p (probability)', 'id': 'p', 'default': 0.5}
    ],
    'Binomial': [
        {'name': 'n (trials)', 'id': 'n', 'default': 10},
        {'name': 'p (probability)', 'id': 'p', 'default': 0.5}
    ],
    'Negative Binomial': [
        {'name': 'r (successes)', 'id': 'r', 'default': 5},
        {'name': 'p (probability)', 'id': 'p', 'default': 0.5}
    ],
    'Gamma': [
        {'name': 'α (shape)', 'id': 'alpha', 'default': 2},
        {'name': 'β (rate)', 'id': 'beta', 'default': 1}
    ],
    'Beta': [
        {'name': 'α (shape)', 'id': 'alpha', 'default': 2},
        {'name': 'β (shape)', 'id': 'beta', 'default': 2}
    ],
    'Chi-square': [
        {'name': 'k (degrees of freedom)', 'id': 'df', 'default': 3}
    ],
    'Pareto': [
        {'name': 'α (shape)', 'id': 'alpha', 'default': 3},
        {'name': 'xₘ (scale)', 'id': 'xm', 'default': 1}
    ],
    'Student\'s T': [
        {'name': 'ν (degrees of freedom)', 'id': 'df', 'default': 3}
    ],
    'Weibull': [
        {'name': 'k (shape)', 'id': 'k', 'default': 1.5},
        {'name': 'λ (scale)', 'id': 'lambda', 'default': 1}
    ]
}

# ── Styles ──────────────────────────────────────────────────────────────────
CARD = {
    'background': '#f8f9fa',
    'borderRadius': '8px',
    'padding': '16px',
    'marginBottom': '16px',
    'border': '1px solid #dee2e6'
}

STAT_BOX = {
    'display': 'inline-block',
    'background': '#fff',
    'borderRadius': '6px',
    'padding': '10px 18px',
    'marginRight': '10px',
    'marginTop': '8px',
    'border': '1px solid #ced4da',
    'minWidth': '120px',
    'textAlign': 'center'
}

# App layout
app.layout = html.Div([
    html.H1('Probability Distribution Simulator',
            style={'textAlign': 'center', 'marginBottom': '4px'}),
    html.P('Explore distributions, real-world scenarios, and the Central Limit Theorem.',
           style={'textAlign': 'center', 'color': '#6c757d', 'marginBottom': '24px'}),

    # ── Row: selector + description ─────────────────────────────────────────
    html.Div([
        # Left: distribution picker & formula
        html.Div([
            html.Div([
                html.Label('Distribution', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='distribution-dropdown',
                    options=[{'label': d, 'value': d} for d in distribution_formulas],
                    value='Normal',
                    clearable=False
                ),
            ], style={'marginBottom': '12px'}),

            html.Div([
                html.Label('Formula', style={'fontWeight': 'bold'}),
                dcc.Markdown(id='formula-display', mathjax=True)
            ]),
        ], style={**CARD, 'flex': '1', 'marginRight': '16px'}),

        # Right: description + scenarios
        html.Div([
            html.Label('What is it?', style={'fontWeight': 'bold'}),
            html.P(id='dist-description',
                   style={'color': '#495057', 'marginBottom': '12px', 'fontSize': '14px'}),
            html.Label('Real-world scenario presets', style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='scenario-dropdown', clearable=True,
                         placeholder='Select a scenario to auto-fill parameters…'),
        ], style={**CARD, 'flex': '1'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),

    # ── Parameters ──────────────────────────────────────────────────────────
    html.Div([
        html.Label('Parameters', style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
        html.Div(id='params-container', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px'}),
    ], style=CARD),

    # ── Key Statistics ───────────────────────────────────────────────────────
    html.Div([
        html.Label('Key Statistics', style={'fontWeight': 'bold'}),
        html.Div(id='stats-panel'),
    ], style=CARD),

    # ── Probability Calculator ───────────────────────────────────────────────
    html.Div([
        html.Label('Probability Calculator', style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
        html.Div([
            html.Div([
                dcc.RadioItems(
                    id='prob-mode',
                    options=[
                        {'label': ' P(X ≤ x)', 'value': 'leq'},
                        {'label': ' P(X ≥ x)', 'value': 'geq'},
                        {'label': ' P(a ≤ X ≤ b)', 'value': 'interval'},
                    ],
                    value='leq',
                    inline=True,
                    style={'marginBottom': '10px'}
                ),
            ]),
            html.Div([
                html.Div([
                    html.Label('x value', style={'fontSize': '13px'}),
                    dcc.Input(id='prob-x1', type='number', value=0, step=0.1,
                              style={'width': '100px', 'marginLeft': '8px'}),
                ], id='prob-x1-wrapper'),
                html.Div([
                    html.Label('Upper bound b', style={'fontSize': '13px'}),
                    dcc.Input(id='prob-x2', type='number', value=1, step=0.1,
                              style={'width': '100px', 'marginLeft': '8px'}),
                ], id='prob-x2-wrapper', style={'marginLeft': '20px', 'display': 'none'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div(id='prob-result',
                     style={'marginTop': '10px', 'fontSize': '20px', 'fontWeight': 'bold', 'color': '#0d6efd'}),
        ])
    ], style=CARD),

    # ── Main Plots ───────────────────────────────────────────────────────────
    html.Div([
        dcc.Graph(id='pdf-pmf-plot', style={'flex': '1'}),
        dcc.Graph(id='cdf-cmf-plot', style={'flex': '1'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px'}),

    # ── Sampling Distribution (CLT) ──────────────────────────────────────────
    html.Div([
        html.H2('Central Limit Theorem Explorer', style={'marginTop': '0'}),
        html.P('Watch how the distribution of sample means becomes Normal as the sample size grows — regardless of the underlying distribution.',
               style={'color': '#6c757d', 'fontSize': '14px'}),
        html.Div([
            html.Div([
                html.Label('Number of samples (N)'),
                dcc.Input(id='sample-size-input', type='number',
                          value=1000, min=100, max=10000, step=100),
            ], style={'marginRight': '24px'}),
            html.Div([
                html.Label('Sample group size (n)'),
                dcc.Input(id='group-size-input', type='number',
                          value=30, min=1, max=100, step=1),
            ]),
            html.Button('Generate Sampling Distribution',
                        id='generate-sampling-button', n_clicks=0,
                        style={'marginLeft': '24px', 'alignSelf': 'flex-end'}),
        ], style={'display': 'flex', 'alignItems': 'flex-end', 'marginBottom': '16px'}),
        dcc.Graph(id='sampling-dist-plot')
    ], style=CARD),

], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '24px'})


# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    [Output('formula-display', 'children'),
     Output('dist-description', 'children'),
     Output('scenario-dropdown', 'options'),
     Output('scenario-dropdown', 'value')],
    Input('distribution-dropdown', 'value')
)
def update_distribution_info(distribution):
    formula = f'${distribution_formulas[distribution]}$'
    description = distribution_descriptions[distribution]
    scenarios = distribution_scenarios[distribution]
    options = [{'label': s['label'], 'value': i} for i, s in enumerate(scenarios)]
    return formula, description, options, None


@app.callback(
    Output('params-container', 'children'),
    [Input('distribution-dropdown', 'value'),
     Input('scenario-dropdown', 'value')]
)
def update_params(distribution, scenario_idx):
    params = distribution_params[distribution]
    # If a scenario is selected, use its parameter values
    preset = {}
    if scenario_idx is not None:
        preset = distribution_scenarios[distribution][int(scenario_idx)]['params']

    return [
        html.Div([
            html.Label(param['name'], style={'fontSize': '13px', 'marginBottom': '4px', 'display': 'block'}),
            dcc.Input(
                id={'type': 'dynamic-param', 'index': param['id']},
                type='number',
                value=preset.get(param['id'], param['default']),
                step=0.1
            )
        ]) for param in params
    ]


@app.callback(
    Output('prob-x2-wrapper', 'style'),
    Input('prob-mode', 'value')
)
def toggle_prob_x2(mode):
    if mode == 'interval':
        return {'marginLeft': '20px', 'display': 'flex', 'alignItems': 'center'}
    return {'marginLeft': '20px', 'display': 'none'}


def get_x_range(dist_name, params):
    if dist_name == 'Uniform':
        a, b = params
        margin = (b - a) * 0.1
        return np.linspace(a - margin, b + margin, 200)
    elif dist_name == 'Exponential':
        rate = params[0]
        mean = 1 / rate
        return np.linspace(0, mean * 5, 200)
    elif dist_name == 'Normal':
        mu, sigma = params
        return np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    elif dist_name == 'Poisson':
        lambda_param = params[0]
        return np.arange(0, max(20, int(lambda_param * 2 + 4 * np.sqrt(lambda_param))))
    elif dist_name == 'Geometric':
        p = params[0]
        mean = 1 / p
        return np.arange(1, int(mean + 4 * np.sqrt((1 - p) / p ** 2)))
    elif dist_name == 'Binomial':
        n, p = params
        mean = n * p
        std = np.sqrt(n * p * (1 - p))
        return np.arange(max(0, int(mean - 3 * std)), min(n, int(mean + 3 * std)) + 1)
    elif dist_name == 'Negative Binomial':
        r, p = params
        mean = r * (1 - p) / p
        std = np.sqrt(r * (1 - p) / p ** 2)
        return np.arange(0, int(mean + 4 * std))
    elif dist_name == 'Gamma':
        alpha, beta = params
        mean = alpha / beta
        std = np.sqrt(alpha / beta ** 2)
        return np.linspace(0, mean + 4 * std, 200)
    elif dist_name == 'Beta':
        return np.linspace(0, 1, 200)
    elif dist_name == 'Chi-square':
        df = params[0]
        return np.linspace(0, max(20, df + 4 * np.sqrt(2 * df)), 200)
    elif dist_name == 'Pareto':
        alpha, xm = params
        if alpha > 1:
            mean = (alpha * xm) / (alpha - 1)
            if alpha > 2:
                std = xm * alpha / ((alpha - 1) * np.sqrt((alpha - 2)))
                return np.linspace(xm, mean + 4 * std, 200)
        return np.linspace(xm, xm * 10, 200)
    elif dist_name == 'Student\'s T':
        df = params[0]
        if df > 2:
            std = np.sqrt(df / (df - 2))
            return np.linspace(-4 * std, 4 * std, 200)
        return np.linspace(-10, 10, 200)
    elif dist_name == 'Weibull':
        k, lambda_param = params
        mean = lambda_param * math.gamma(1 + 1 / k)
        var = lambda_param ** 2 * (math.gamma(1 + 2 / k) - (math.gamma(1 + 1 / k)) ** 2)
        std = np.sqrt(var)
        return np.linspace(0, mean + 4 * std, 200)


def build_dist(distribution, param_values):
    """Return a scipy.stats frozen distribution or raise ValueError."""
    if distribution == 'Uniform':
        a, b = param_values
        if b <= a:
            raise ValueError("b must be > a")
        return stats.uniform(loc=a, scale=b - a)
    elif distribution == 'Exponential':
        rate = param_values[0]
        if rate <= 0:
            raise ValueError("λ must be positive")
        return stats.expon(scale=1 / rate)
    elif distribution == 'Normal':
        mu, sigma = param_values
        return stats.norm(mu, sigma)
    elif distribution == 'Poisson':
        return stats.poisson(param_values[0])
    elif distribution == 'Geometric':
        return stats.geom(param_values[0])
    elif distribution == 'Binomial':
        n, p = param_values
        return stats.binom(int(n), p)
    elif distribution == 'Negative Binomial':
        r, p = param_values
        return stats.nbinom(int(r), p)
    elif distribution == 'Gamma':
        alpha, beta = param_values
        return stats.gamma(alpha, scale=1 / beta)
    elif distribution == 'Beta':
        alpha, beta = param_values
        return stats.beta(alpha, beta)
    elif distribution == 'Chi-square':
        return stats.chi2(param_values[0])
    elif distribution == 'Pareto':
        alpha, xm = param_values
        return stats.pareto(alpha, scale=xm)
    elif distribution == 'Student\'s T':
        return stats.t(param_values[0])
    elif distribution == 'Weibull':
        k, lambda_param = param_values
        return stats.weibull_min(k, scale=lambda_param)


def fmt(v):
    """Format a float nicely."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return '—'
    return f'{v:.4g}'


@app.callback(
    [Output('pdf-pmf-plot', 'figure'),
     Output('cdf-cmf-plot', 'figure'),
     Output('stats-panel', 'children'),
     Output('prob-result', 'children')],
    [Input('distribution-dropdown', 'value'),
     Input({'type': 'dynamic-param', 'index': ALL}, 'value'),
     Input('prob-mode', 'value'),
     Input('prob-x1', 'value'),
     Input('prob-x2', 'value')]
)
def update_main(distribution, param_values, prob_mode, prob_x1, prob_x2):
    dist_params = distribution_params[distribution]
    param_values = param_values[:len(dist_params)]

    if any(v is None for v in param_values):
        empty = {'data': [], 'layout': {'title': 'Waiting for valid parameters…'}}
        return empty, empty, '', ''

    x = get_x_range(distribution, param_values)
    discrete = distribution in ['Poisson', 'Geometric', 'Binomial', 'Negative Binomial']

    try:
        dist = build_dist(distribution, param_values)

        # ── PDF/PMF ──────────────────────────────────────────────────────────
        shade_color = 'rgba(13, 110, 253, 0.2)'
        border_color = '#0d6efd'

        if discrete:
            pmf = dist.pmf(x)
            # Shade selected region
            shade_x, shade_y = [], []
            if prob_x1 is not None:
                for xi, yi in zip(x, pmf):
                    include = False
                    if prob_mode == 'leq' and xi <= prob_x1:
                        include = True
                    elif prob_mode == 'geq' and xi >= prob_x1:
                        include = True
                    elif prob_mode == 'interval' and prob_x2 is not None and prob_x1 <= xi <= prob_x2:
                        include = True
                    if include:
                        shade_x.append(xi)
                        shade_y.append(yi)

            traces = [go.Bar(x=x, y=pmf, name='PMF', marker_color='#6c757d', opacity=0.6)]
            if shade_x:
                traces.append(go.Bar(x=shade_x, y=shade_y, name='Selected region',
                                     marker_color=border_color))
            pdf_pmf_fig = go.Figure(traces)
            pdf_pmf_fig.update_layout(
                title=f'{distribution} PMF', barmode='overlay',
                xaxis_title='k', yaxis_title='P(X = k)',
                yaxis_range=[0, max(pmf) * 1.15], legend=dict(orientation='h'))

        else:
            pdf = dist.pdf(x)
            # Build shaded fill trace
            fill_x, fill_y = [], []
            if prob_x1 is not None:
                for xi, yi in zip(x, pdf):
                    include = False
                    if prob_mode == 'leq' and xi <= prob_x1:
                        include = True
                    elif prob_mode == 'geq' and xi >= prob_x1:
                        include = True
                    elif prob_mode == 'interval' and prob_x2 is not None and prob_x1 <= xi <= prob_x2:
                        include = True
                    if include:
                        fill_x.append(xi)
                        fill_y.append(yi)

            traces = [go.Scatter(x=x, y=pdf, mode='lines', name='PDF',
                                 line=dict(color='#6c757d', width=2))]
            if fill_x:
                traces.append(go.Scatter(
                    x=fill_x + fill_x[::-1],
                    y=fill_y + [0] * len(fill_y),
                    fill='toself', fillcolor=shade_color,
                    line=dict(color=border_color, width=1),
                    name='Selected region'))

            pdf_pmf_fig = go.Figure(traces)
            pdf_pmf_fig.update_layout(
                title=f'{distribution} PDF',
                xaxis_title='x', yaxis_title='f(x)',
                yaxis_range=[0, max(pdf) * 1.15], legend=dict(orientation='h'))

        # ── CDF ──────────────────────────────────────────────────────────────
        cdf = dist.cdf(x)
        cdf_fig = go.Figure([go.Scatter(x=x, y=cdf, mode='lines', name='CDF',
                                        line=dict(color='#fd7e14', width=2))])
        cdf_fig.update_layout(title=f'{distribution} CDF',
                               xaxis_title='x', yaxis_title='F(x)',
                               yaxis_range=[0, 1.05])

        # ── Key Statistics ────────────────────────────────────────────────────
        try:
            mean = dist.mean()
            variance = dist.var()
            std = dist.std()
            skew = dist.stats(moments='s')
            median = dist.median()
        except Exception:
            mean = variance = std = skew = median = None

        stats_children = html.Div([
            html.Div([html.Div('Mean', style={'fontSize': '11px', 'color': '#6c757d'}),
                      html.Div(fmt(mean), style={'fontSize': '18px', 'fontWeight': 'bold'})],
                     style=STAT_BOX),
            html.Div([html.Div('Median', style={'fontSize': '11px', 'color': '#6c757d'}),
                      html.Div(fmt(median), style={'fontSize': '18px', 'fontWeight': 'bold'})],
                     style=STAT_BOX),
            html.Div([html.Div('Std Dev', style={'fontSize': '11px', 'color': '#6c757d'}),
                      html.Div(fmt(std), style={'fontSize': '18px', 'fontWeight': 'bold'})],
                     style=STAT_BOX),
            html.Div([html.Div('Variance', style={'fontSize': '11px', 'color': '#6c757d'}),
                      html.Div(fmt(variance), style={'fontSize': '18px', 'fontWeight': 'bold'})],
                     style=STAT_BOX),
            html.Div([html.Div('Skewness', style={'fontSize': '11px', 'color': '#6c757d'}),
                      html.Div(fmt(float(skew)) if skew is not None else '—',
                               style={'fontSize': '18px', 'fontWeight': 'bold'})],
                     style=STAT_BOX),
        ])

        # ── Probability Result ────────────────────────────────────────────────
        prob_text = ''
        if prob_x1 is not None:
            try:
                if prob_mode == 'leq':
                    p = dist.cdf(prob_x1)
                    prob_text = f'P(X ≤ {prob_x1}) = {p:.4f}'
                elif prob_mode == 'geq':
                    p = 1 - dist.cdf(prob_x1)
                    if discrete:
                        p = 1 - dist.cdf(prob_x1 - 1)
                    prob_text = f'P(X ≥ {prob_x1}) = {p:.4f}'
                elif prob_mode == 'interval' and prob_x2 is not None:
                    if discrete:
                        p = dist.cdf(prob_x2) - dist.cdf(prob_x1 - 1)
                    else:
                        p = dist.cdf(prob_x2) - dist.cdf(prob_x1)
                    prob_text = f'P({prob_x1} ≤ X ≤ {prob_x2}) = {p:.4f}'
            except Exception:
                prob_text = 'Invalid range'

        return pdf_pmf_fig, cdf_fig, stats_children, prob_text

    except Exception as e:
        err = {'data': [], 'layout': {'title': f'Error: {e}'}}
        return err, err, f'Error: {e}', ''


@app.callback(
    Output('sampling-dist-plot', 'figure'),
    Input('generate-sampling-button', 'n_clicks'),
    [State('distribution-dropdown', 'value'),
     State({'type': 'dynamic-param', 'index': ALL}, 'value'),
     State('sample-size-input', 'value'),
     State('group-size-input', 'value')]
)
def update_sampling(n_clicks, distribution, param_values, sample_size, group_size):
    if n_clicks == 0:
        return {}

    dist_params = distribution_params[distribution]
    param_values = param_values[:len(dist_params)]

    try:
        dist = build_dist(distribution, param_values)
        random_samples = dist.rvs(size=(sample_size, group_size))
        sample_means = np.mean(random_samples, axis=1)

        means_mean = np.mean(sample_means)
        means_std = np.std(sample_means)
        x_ref = np.linspace(means_mean - 4 * means_std, means_mean + 4 * means_std, 100)
        normal_ref = stats.norm.pdf(x_ref, means_mean, means_std)

        fig = go.Figure([
            go.Histogram(x=sample_means, name='Sample Means',
                         histnorm='probability density', opacity=0.7, nbinsx=40),
            go.Scatter(x=x_ref, y=normal_ref, name='Normal Reference',
                       line=dict(color='red', dash='dash', width=2))
        ])
        fig.update_layout(
            title=f'Sampling Distribution of Means  (N={sample_size}, n={group_size})',
            xaxis_title='Sample Mean', yaxis_title='Density',
            showlegend=True, bargap=0.05,
            legend=dict(orientation='h')
        )
        return fig
    except Exception as e:
        return {'data': [], 'layout': {'title': f'Error: {e}'}}


if __name__ == '__main__':
    app.run_server(debug=True)
