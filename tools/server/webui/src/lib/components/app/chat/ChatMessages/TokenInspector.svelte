<script lang="ts">
	import { ChevronDown, ChevronUp, Pin } from '@lucide/svelte';

	export interface TokenLogprobData {
		token: string;
		logprob: number;
		top_logprobs: Array<{ token: string; logprob: number }>;
	}

	interface Props {
		tokens: TokenLogprobData[];
	}

	let { tokens }: Props = $props();
	let expanded = $state(false);

	function computeEntropy(data: TokenLogprobData): number {
		let h = 0;
		const items =
			data.top_logprobs.length > 0
				? data.top_logprobs
				: [{ token: data.token, logprob: data.logprob }];
		for (const t of items) {
			const p = Math.exp(t.logprob);
			if (p > 0 && p <= 1) {
				h -= p * Math.log2(p);
			}
		}
		return h;
	}

	function formatToken(t: string): string {
		if (t === ' ') return '\u2423'; // ␡ for space
		return t.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
	}

	let displayTokens = $derived(tokens.slice(0, 100));
	let hasMore = $derived(tokens.length > 100);

	// Popup state: follows mouse cursor like Google Maps
	let activePopup = $state<{
		idx: number;
		x: number;
		y: number;
		pinned: boolean;
	} | null>(null);
	let popupHoverActive = $state(false);

	function showPopup(e: MouseEvent, idx: number) {
		if (popupHoverActive && activePopup?.pinned) return;
		activePopup = { idx, x: e.clientX, y: e.clientY, pinned: false };
		popupHoverActive = true;
	}

	function movePopup(e: MouseEvent) {
		if (!popupHoverActive || !activePopup || activePopup.pinned) return;
		activePopup = { ...activePopup, x: e.clientX, y: e.clientY };
	}

	function hidePopup() {
		popupHoverActive = false;
		if (activePopup && !activePopup.pinned) {
			activePopup = null;
		}
	}

	function togglePin() {
		if (activePopup) {
			const newPinned = !activePopup.pinned;
			activePopup = { ...activePopup, pinned: newPinned };
			if (!newPinned && !popupHoverActive) {
				activePopup = null;
			}
		}
	}
</script>

<div class="mt-2 rounded-md border border-border bg-muted/30">
	<button
		type="button"
		onclick={() => (expanded = !expanded)}
		class="flex w-full items-center justify-between rounded-md px-3 py-2 text-sm hover:bg-muted/50"
	>
		<div class="flex items-center gap-2">
			{#if expanded}
				<ChevronDown class="h-4 w-4" />
			{:else}
				<ChevronUp class="h-4 w-4" />
			{/if}
			<span class="font-medium">Token Inspection</span>
			<span class="text-xs text-muted-foreground">
				{tokens.length} tokens
			</span>
		</div>
	</button>

	{#if expanded}
		<div
			class="max-h-96 overflow-y-auto px-3 pb-3"
			onmousemove={movePopup}
			onmouseleave={hidePopup}
			role="region"
			aria-label="Token list"
		>
			<div class="flex flex-wrap gap-1">
				{#each displayTokens as t, idx (t.token + idx)}
					{@const mainProb = Math.exp(t.logprob)}
					{@const ent = computeEntropy(t)}
					<button
						type="button"
						onclick={(e) => {
							e.stopPropagation();
							showPopup(e, idx);
						}}
						onmouseenter={(e) => showPopup(e, idx)}
						onmouseleave={hidePopup}
						class={`
							cursor-pointer rounded px-1.5 py-0.5 font-mono text-[11px]
							leading-snug transition-colors
							${mainProb >= 0.7 ? 'bg-green-800/70 text-green-50 dark:bg-green-800/90 dark:text-green-100' : ''}
							${mainProb >= 0.3 && mainProb < 0.7 ? 'bg-yellow-800/70 text-yellow-100 dark:bg-yellow-800/90 dark:text-yellow-100' : ''}
							${mainProb < 0.3 ? 'bg-red-800/70 text-white dark:bg-red-800/90 dark:text-red-50' : ''}
							${activePopup?.idx === idx ? 'ring-2 ring-blue-500' : ''}
							hover:brightness-125
						`}
						title="{formatToken(t.token)} — ${(mainProb * 100).toFixed(1)}% — H={ent.toFixed(1)}"
					>
						{formatToken(t.token)}
					</button>
				{/each}
			</div>
			{#if hasMore}
				<div class="mt-2 text-center text-xs text-muted-foreground">
					+{tokens.length - 100} more tokens (not shown)
				</div>
			{/if}

			<!-- Summary bar at bottom -->
			<div class="mt-3 flex items-center justify-between text-[10px] text-muted-foreground">
				<div class="flex items-center gap-3">
					<span class="flex items-center gap-1">
						<span class="inline-block h-2.5 w-2.5 rounded bg-green-800/70 dark:bg-green-800/90"
						></span>
						High (≥70%)
					</span>
					<span class="flex items-center gap-1">
						<span class="inline-block h-2.5 w-2.5 rounded bg-yellow-800/70 dark:bg-yellow-800/90"
						></span>
						Medium (30-69%)
					</span>
					<span class="flex items-center gap-1">
						<span class="inline-block h-2.5 w-2.5 rounded bg-red-800/70 dark:bg-red-800/90"></span>
						Low (&lt;30%)
					</span>
				</div>
				<span>Click token for details. Hover to preview.</span>
			</div>
		</div>
	{/if}
</div>

<!-- Alternative Token Popup -->
{#if activePopup !== null}
	{@const t = displayTokens[activePopup.idx]}
	{@const mainProb = Math.exp(t.logprob)}
	{@const ent = computeEntropy(t)}

	<div
		class="fixed z-50 rounded-lg border border-border bg-popover px-3.5 py-3 text-xs shadow-xl"
		style="left: {activePopup.x}px; top: {activePopup.y}px; transform: translate(-50%, -100%);"
		onmouseleave={() => {
			if (!activePopup?.pinned) activePopup = null;
		}}
	>
		<div class="mb-2 flex items-center justify-between">
			<div class="flex items-center gap-2">
				<span class="font-mono text-base font-bold">{formatToken(t.token)}</span>
				<span
					class={`rounded px-2 py-0.5 text-xs font-semibold text-white
						${mainProb >= 0.7 ? 'bg-green-700 dark:bg-green-700' : ''}
						${mainProb >= 0.3 && mainProb < 0.7 ? 'bg-yellow-600 dark:bg-yellow-600' : ''}
						${mainProb < 0.3 ? 'bg-red-600 dark:bg-red-600' : ''}`}
				>
					{(mainProb * 100).toFixed(1)}%
				</span>
			</div>
			<button
				onclick={togglePin}
				class="rounded p-1 text-muted-foreground hover:bg-muted/50 hover:text-foreground"
			>
				<Pin class="h-3.5 w-3.5" />
				{#if activePopup.pinned}
					<span class="text-blue-500">✓</span>
				{/if}
			</button>
		</div>
		<div class="grid grid-cols-2 gap-x-6 text-muted-foreground">
			<div>Logprob: <span class="text-foreground tabular-nums">{t.logprob.toFixed(3)}</span></div>
			<div>
				Entropy: <span
					class="tabular-nums {ent < 1
						? 'text-green-500'
						: ent < 3
							? 'text-yellow-500'
							: 'text-red-500'}">{ent.toFixed(2)}</span
				>
			</div>
		</div>

		{#if t.top_logprobs.length > 0}
			<div class="mt-2 border-t border-border pt-2">
				<div class="mb-1 text-[10px] font-medium tracking-wider text-muted-foreground uppercase">
					Alternative Tokens
				</div>
				{#each t.top_logprobs.slice(0, 8) as alt, i (alt.token + i)}
					{@const altProb = Math.exp(alt.logprob)}
					<div class="mb-0.5 flex items-center gap-1.5 text-[11px]">
						<span class="w-4 text-right text-muted-foreground/60">#{i + 1}</span>
						<span class="w-20 truncate font-mono">{formatToken(alt.token)}</span>
						<div class="h-2.5 flex-1 overflow-hidden rounded-sm bg-muted/50">
							<div
								class="h-full rounded-sm
									{altProb >= 0.5 ? 'bg-green-500 dark:bg-green-600' : ''}
									{altProb >= 0.1 && altProb < 0.5 ? 'bg-yellow-500 dark:bg-yellow-600' : ''}
									{altProb < 0.1 ? 'bg-red-400 dark:bg-red-500' : ''}"
								style="width: {Math.min(altProb * 100, 100)}%;"
							></div>
						</div>
						<span class="w-10 text-right text-muted-foreground tabular-nums">
							{(altProb * 100).toFixed(1)}%
						</span>
						<span class="w-14 text-right text-[10px] text-muted-foreground/60 tabular-nums">
							{alt.logprob.toFixed(2)}
						</span>
					</div>
				{/each}
			</div>
		{/if}
	</div>
{/if}
